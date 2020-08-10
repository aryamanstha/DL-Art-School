import torch
from torch import nn
from switched_conv import BareConvSwitch, compute_attention_specificity, AttentionNorm
import torch.nn.functional as F
import functools
from collections import OrderedDict
from models.archs.arch_util import ConvBnLelu, ConvGnSilu, ExpansionBlock
from models.archs.RRDBNet_arch import ResidualDenseBlock_5C, RRDB
from models.archs.spinenet_arch import SpineNet
from switched_conv_util import save_attention_to_image_rgb
import os


class MultiConvBlock(nn.Module):
    def __init__(self, filters_in, filters_mid, filters_out, kernel_size, depth, scale_init=1, norm=False, weight_init_factor=1):
        assert depth >= 2
        super(MultiConvBlock, self).__init__()
        self.noise_scale = nn.Parameter(torch.full((1,), fill_value=.01))
        self.bnconvs = nn.ModuleList([ConvBnLelu(filters_in, filters_mid, kernel_size, norm=norm, bias=False, weight_init_factor=weight_init_factor)] +
                                     [ConvBnLelu(filters_mid, filters_mid, kernel_size, norm=norm, bias=False, weight_init_factor=weight_init_factor) for i in range(depth - 2)] +
                                     [ConvBnLelu(filters_mid, filters_out, kernel_size, activation=False, norm=False, bias=False, weight_init_factor=weight_init_factor)])
        self.scale = nn.Parameter(torch.full((1,), fill_value=scale_init, dtype=torch.float))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is not None:
            noise = noise * self.noise_scale
            x = x + noise
        for m in self.bnconvs:
            x = m.forward(x)
        return x * self.scale + self.bias


# VGG-style layer with Conv(stride2)->BN->Activation->Conv->BN->Activation
# Doubles the input filter count.
class HalvingProcessingBlock(nn.Module):
    def __init__(self, filters):
        super(HalvingProcessingBlock, self).__init__()
        self.bnconv1 = ConvGnSilu(filters, filters * 2, stride=2, norm=False, bias=False)
        self.bnconv2 = ConvGnSilu(filters * 2, filters * 2, norm=True, bias=False)

    def forward(self, x):
        x = self.bnconv1(x)
        return self.bnconv2(x)


# This is a classic u-net architecture with the goal of assigning each individual pixel an individual transform
# switching set.
class ConvBasisMultiplexer(nn.Module):
    def __init__(self, input_channels, base_filters, reductions, processing_depth, multiplexer_channels, use_gn=True):
        super(ConvBasisMultiplexer, self).__init__()
        self.filter_conv = ConvGnSilu(input_channels, base_filters, bias=True)
        self.reduction_blocks = nn.ModuleList([HalvingProcessingBlock(base_filters * 2 ** i) for i in range(reductions)])
        reduction_filters = base_filters * 2 ** reductions
        self.processing_blocks = nn.Sequential(OrderedDict([('block%i' % (i,), ConvGnSilu(reduction_filters, reduction_filters, bias=False)) for i in range(processing_depth)]))
        self.expansion_blocks = nn.ModuleList([ExpansionBlock(reduction_filters // (2 ** i)) for i in range(reductions)])

        gap = base_filters - multiplexer_channels
        cbl1_out = ((base_filters - (gap // 2)) // 4) * 4   # Must be multiples of 4 to use with group norm.
        self.cbl1 = ConvGnSilu(base_filters, cbl1_out, norm=use_gn, bias=False, num_groups=4)
        cbl2_out = ((base_filters - (3 * gap // 4)) // 4) * 4
        self.cbl2 = ConvGnSilu(cbl1_out, cbl2_out, norm=use_gn, bias=False, num_groups=4)
        self.cbl3 = ConvGnSilu(cbl2_out, multiplexer_channels, bias=True, norm=False)

    def forward(self, x):
        x = self.filter_conv(x)
        reduction_identities = []
        for b in self.reduction_blocks:
            reduction_identities.append(x)
            x = b(x)
        x = self.processing_blocks(x)
        for i, b in enumerate(self.expansion_blocks):
            x = b(x, reduction_identities[-i - 1])

        x = self.cbl1(x)
        x = self.cbl2(x)
        x = self.cbl3(x)
        return x


class CachedBackboneWrapper:
    def __init__(self, backbone: nn.Module):
        self.backbone = backbone

    def __call__(self, *args):
        self.cache = self.backbone(*args)
        return self.cache

    def get_forward_result(self):
        return self.cache


class BackboneMultiplexer(nn.Module):
    def __init__(self, backbone: CachedBackboneWrapper, transform_count):
        super(BackboneMultiplexer, self).__init__()
        self.backbone = backbone
        self.proc = nn.Sequential(ConvGnSilu(256, 256, kernel_size=3, bias=True),
                                  ConvGnSilu(256, 256, kernel_size=3, bias=False))
        self.up1 = nn.Sequential(ConvGnSilu(256, 128, kernel_size=3, bias=False, norm=False, activation=False),
                                 ConvGnSilu(128, 128, kernel_size=3, bias=False))
        self.up2 = nn.Sequential(ConvGnSilu(128, 64, kernel_size=3, bias=False, norm=False, activation=False),
                                 ConvGnSilu(64, 64, kernel_size=3, bias=False))
        self.final = ConvGnSilu(64, transform_count, bias=False, norm=False, activation=False)

    def forward(self, x):
        spine = self.backbone.get_forward_result()
        feat = self.proc(spine[0])
        feat = self.up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        feat = self.up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        return self.final(feat)


class ConfigurableSwitchComputer(nn.Module):
    def __init__(self, base_filters, multiplexer_net, pre_transform_block, transform_block, transform_count, attention_norm,
                 init_temp=20, add_scalable_noise_to_transforms=False):
        super(ConfigurableSwitchComputer, self).__init__()

        tc = transform_count
        self.multiplexer = multiplexer_net(tc)

        if pre_transform_block:
            self.pre_transform = pre_transform_block()
        else:
            self.pre_transform = None
        self.transforms = nn.ModuleList([transform_block() for _ in range(transform_count)])
        self.add_noise = add_scalable_noise_to_transforms
        self.noise_scale = nn.Parameter(torch.full((1,), float(1e-3)))

        # And the switch itself, including learned scalars
        self.switch = BareConvSwitch(initial_temperature=init_temp, attention_norm=AttentionNorm(transform_count, accumulator_size=16 * transform_count) if attention_norm else None)
        self.switch_scale = nn.Parameter(torch.full((1,), float(1)))
        self.post_switch_conv = ConvBnLelu(base_filters, base_filters, norm=False, bias=True)
        # The post_switch_conv gets a low scale initially. The network can decide to magnify it (or not)
        # depending on its needs.
        self.psc_scale = nn.Parameter(torch.full((1,), float(.1)))

    def forward(self, x, output_attention_weights=False, identity=None, att_in=None, fixed_scale=1):
        if att_in is None:
            att_in = x

        if identity is None:
            identity = x

        if self.add_noise:
            rand_feature = torch.randn_like(x) * self.noise_scale
            x = x + rand_feature

        if self.pre_transform:
            x = self.pre_transform(x)
        xformed = [t.forward(x) for t in self.transforms]
        m = self.multiplexer(att_in)


        outputs, attention = self.switch(xformed, m, True)
        outputs = identity + outputs * self.switch_scale * fixed_scale
        outputs = outputs + self.post_switch_conv(outputs) * self.psc_scale * fixed_scale
        if output_attention_weights:
            return outputs, attention
        else:
            return outputs

    def set_temperature(self, temp):
        self.switch.set_attention_temperature(temp)


class ConfigurableSwitchedResidualGenerator2(nn.Module):
    def __init__(self, switch_depth, switch_filters, switch_reductions, switch_processing_layers, trans_counts, trans_kernel_sizes,
                 trans_layers, transformation_filters, attention_norm, initial_temp=20, final_temperature_step=50000, heightened_temp_min=1,
                 heightened_final_step=50000, upsample_factor=1,
                 add_scalable_noise_to_transforms=False):
        super(ConfigurableSwitchedResidualGenerator2, self).__init__()
        switches = []
        self.initial_conv = ConvBnLelu(3, transformation_filters, norm=False, activation=False, bias=True)
        self.upconv1 = ConvBnLelu(transformation_filters, transformation_filters, norm=False, bias=True)
        self.upconv2 = ConvBnLelu(transformation_filters, transformation_filters, norm=False, bias=True)
        self.hr_conv = ConvBnLelu(transformation_filters, transformation_filters, norm=False, bias=True)
        self.final_conv = ConvBnLelu(transformation_filters, 3, norm=False, activation=False, bias=True)
        for _ in range(switch_depth):
            multiplx_fn = functools.partial(ConvBasisMultiplexer, transformation_filters, switch_filters, switch_reductions, switch_processing_layers, trans_counts)
            pretransform_fn = functools.partial(ConvBnLelu, transformation_filters, transformation_filters, norm=False, bias=False, weight_init_factor=.1)
            transform_fn = functools.partial(MultiConvBlock, transformation_filters, int(transformation_filters * 1.5), transformation_filters, kernel_size=trans_kernel_sizes, depth=trans_layers, weight_init_factor=.1)
            switches.append(ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                       pre_transform_block=pretransform_fn, transform_block=transform_fn,
                                                       attention_norm=attention_norm,
                                                       transform_count=trans_counts, init_temp=initial_temp,
                                                       add_scalable_noise_to_transforms=add_scalable_noise_to_transforms))

        self.switches = nn.ModuleList(switches)
        self.transformation_counts = trans_counts
        self.init_temperature = initial_temp
        self.final_temperature_step = final_temperature_step
        self.heightened_temp_min = heightened_temp_min
        self.heightened_final_step = heightened_final_step
        self.attentions = None
        self.upsample_factor = upsample_factor
        assert self.upsample_factor == 2 or self.upsample_factor == 4

    def forward(self, x):
        # This is a common bug when evaluating SRG2 generators. It needs to be configured properly in eval mode. Just fail.
        if not self.train:
            assert self.switches[0].switch.temperature == 1

        x = self.initial_conv(x)

        self.attentions = []
        for i, sw in enumerate(self.switches):
            x, att = sw.forward(x, True)
            self.attentions.append(att)

        x = self.upconv1(F.interpolate(x, scale_factor=2, mode="nearest"))
        if self.upsample_factor > 2:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.upconv2(x)
        x = self.final_conv(self.hr_conv(x))
        return x, x

    def set_temperature(self, temp):
        [sw.set_temperature(temp) for sw in self.switches]

    def update_for_step(self, step, experiments_path='.'):
        if self.attentions:
            temp = max(1,
                1 + self.init_temperature * (self.final_temperature_step - step) / self.final_temperature_step)
            if temp == 1 and self.heightened_final_step and step > self.final_temperature_step and \
                    self.heightened_final_step != 1:
                # Once the temperature passes (1) it enters an inverted curve to match the linear curve from above.
                # without this, the attention specificity "spikes" incredibly fast in the last few iterations.
                h_steps_total = self.heightened_final_step - self.final_temperature_step
                h_steps_current = min(step - self.final_temperature_step, h_steps_total)
                # The "gap" will represent the steps that need to be traveled as a linear function.
                h_gap = 1 / self.heightened_temp_min
                temp = h_gap * h_steps_current / h_steps_total
                # Invert temperature to represent reality on this side of the curve
                temp = 1 / temp
            self.set_temperature(temp)
            if step % 50 == 0:
                output_path = os.path.join(experiments_path, "attention_maps", "a%i")
                prefix = "attention_map_%i_%%i.png" % (step,)
                [save_attention_to_image_rgb(output_path % (i,), self.attentions[i], self.transformation_counts, prefix, step) for i in range(len(self.attentions))]

    def get_debug_values(self, step):
        temp = self.switches[0].switch.temperature
        mean_hists = [compute_attention_specificity(att, 2) for att in self.attentions]
        means = [i[0] for i in mean_hists]
        hists = [i[1].clone().detach().cpu().flatten() for i in mean_hists]
        val = {"switch_temperature": temp}
        for i in range(len(means)):
            val["switch_%i_specificity" % (i,)] = means[i]
            val["switch_%i_histogram" % (i,)] = hists[i]
        return val


# Equivalent to SRG2 - Uses RDB blocks in between two switches.
class ConfigurableSwitchedResidualGenerator4(nn.Module):
    def __init__(self, switch_filters, switch_reductions, switch_processing_layers, trans_counts, trans_kernel_sizes,
                 trans_layers, transformation_filters, attention_norm, initial_temp=20, final_temperature_step=50000, heightened_temp_min=1,
                 heightened_final_step=50000, upsample_factor=1,
                 add_scalable_noise_to_transforms=False):
        super(ConfigurableSwitchedResidualGenerator4, self).__init__()
        self.initial_conv = ConvBnLelu(3, transformation_filters, norm=False, activation=False, bias=True)
        self.upconv1 = ConvBnLelu(transformation_filters, transformation_filters, norm=False, bias=True)
        self.upconv2 = ConvBnLelu(transformation_filters, transformation_filters, norm=False, bias=True)
        self.hr_conv = ConvBnLelu(transformation_filters, transformation_filters, norm=False, bias=True)

        multiplx_fn = functools.partial(ConvBasisMultiplexer, transformation_filters, switch_filters, switch_reductions,
                                        switch_processing_layers, trans_counts)
        half_multiplx_fn = functools.partial(ConvBasisMultiplexer, transformation_filters, switch_filters, switch_reductions,
                                        switch_processing_layers, trans_counts // 2)
        transform_fn = functools.partial(MultiConvBlock, transformation_filters, int(transformation_filters * 1.5),
                                         transformation_filters, kernel_size=trans_kernel_sizes, depth=trans_layers,
                                         weight_init_factor=.1)
        self.rdb1 = RRDB(transformation_filters)
        self.sw1 = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=None, transform_block=transform_fn,
                                                   attention_norm=attention_norm,
                                                   transform_count=trans_counts, init_temp=initial_temp,
                                                   add_scalable_noise_to_transforms=add_scalable_noise_to_transforms)
        self.rdb2 = RRDB(transformation_filters)
        self.sw2 = ConfigurableSwitchComputer(transformation_filters, half_multiplx_fn,
                                                   pre_transform_block=None, transform_block=transform_fn,
                                                   attention_norm=attention_norm,
                                                   transform_count=trans_counts // 2, init_temp=initial_temp,
                                                   add_scalable_noise_to_transforms=add_scalable_noise_to_transforms)
        self.rdb3 = RRDB(transformation_filters)
        self.sw3 = ConfigurableSwitchComputer(transformation_filters, multiplx_fn,
                                                   pre_transform_block=None, transform_block=transform_fn,
                                                   attention_norm=attention_norm,
                                                   transform_count=trans_counts, init_temp=initial_temp,
                                                   add_scalable_noise_to_transforms=add_scalable_noise_to_transforms)
        self.rdb4 = RRDB(transformation_filters)
        self.switches = [self.sw1, self.sw2, self.sw3]

        self.final_conv = ConvBnLelu(transformation_filters, 3, norm=False, activation=False, bias=True)
        self.transformation_counts = trans_counts
        self.init_temperature = initial_temp
        self.final_temperature_step = final_temperature_step
        self.heightened_temp_min = heightened_temp_min
        self.heightened_final_step = heightened_final_step
        self.attentions = None
        self.upsample_factor = upsample_factor
        assert self.upsample_factor == 2 or self.upsample_factor == 4

    def forward(self, x):
        # This is a common bug when evaluating SRG2 generators. It needs to be configured properly in eval mode. Just fail.
        if not self.train:
            assert self.switches[0].switch.temperature == 1

        x = self.initial_conv(x)

        x = self.rdb1(x)
        x, a1 = self.sw1(x, True)
        x = self.rdb2(x)
        x, a2 = self.sw2(x, True)
        x = self.rdb3(x)
        x, a3 = self.sw3(x, True)
        x = self.rdb4(x)
        self.attentions = [a1, a2, a3]

        x = self.upconv1(F.interpolate(x, scale_factor=2, mode="nearest"))
        if self.upsample_factor > 2:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.upconv2(x)
        x = self.final_conv(self.hr_conv(x))
        return x, x

    def set_temperature(self, temp):
        [sw.set_temperature(temp) for sw in self.switches]

    def update_for_step(self, step, experiments_path='.'):
        if self.attentions:
            temp = max(1,
                1 + self.init_temperature * (self.final_temperature_step - step) / self.final_temperature_step)
            if temp == 1 and self.heightened_final_step and step > self.final_temperature_step and \
                    self.heightened_final_step != 1:
                # Once the temperature passes (1) it enters an inverted curve to match the linear curve from above.
                # without this, the attention specificity "spikes" incredibly fast in the last few iterations.
                h_steps_total = self.heightened_final_step - self.final_temperature_step
                h_steps_current = min(step - self.final_temperature_step, h_steps_total)
                # The "gap" will represent the steps that need to be traveled as a linear function.
                h_gap = 1 / self.heightened_temp_min
                temp = h_gap * h_steps_current / h_steps_total
                # Invert temperature to represent reality on this side of the curve
                temp = 1 / temp
            self.set_temperature(temp)
            if step % 50 == 0:
                output_path = os.path.join(experiments_path, "attention_maps", "a%i")
                prefix = "attention_map_%i_%%i.png" % (step,)
                [save_attention_to_image_rgb(output_path % (i,), self.attentions[i], self.transformation_counts, prefix, step) for i in range(len(self.attentions))]

    def get_debug_values(self, step):
        temp = self.switches[0].switch.temperature
        mean_hists = [compute_attention_specificity(att, 2) for att in self.attentions]
        means = [i[0] for i in mean_hists]
        hists = [i[1].clone().detach().cpu().flatten() for i in mean_hists]
        val = {"switch_temperature": temp}
        for i in range(len(means)):
            val["switch_%i_specificity" % (i,)] = means[i]
            val["switch_%i_histogram" % (i,)] = hists[i]
        return val

class Interpolate(nn.Module):
    def __init__(self, factor):
        super(Interpolate, self).__init__()
        self.factor = factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.factor)


class ConfigurableSwitchedResidualGenerator3(nn.Module):
    def __init__(self, base_filters, trans_count, initial_temp=20, final_temperature_step=50000,
                 heightened_temp_min=1,
                 heightened_final_step=50000, upsample_factor=4):
        super(ConfigurableSwitchedResidualGenerator3, self).__init__()
        self.initial_conv = ConvBnLelu(3, base_filters, norm=False, activation=False, bias=True)
        self.sw_conv = ConvBnLelu(base_filters, base_filters, activation=False, bias=True)
        self.upconv1 = ConvBnLelu(base_filters, base_filters, norm=False, bias=True)
        self.upconv2 = ConvBnLelu(base_filters, base_filters, norm=False, bias=True)
        self.hr_conv = ConvBnLelu(base_filters, base_filters, norm=False, bias=True)
        self.final_conv = ConvBnLelu(base_filters, 3, norm=False, activation=False, bias=True)

        self.backbone = SpineNet('49', in_channels=3, use_input_norm=True)
        for p in self.backbone.parameters(recurse=True):
            p.requires_grad = False
        self.backbone_wrapper = CachedBackboneWrapper(self.backbone)
        multiplx_fn = functools.partial(BackboneMultiplexer, self.backbone_wrapper)
        pretransform_fn = functools.partial(nn.Sequential, ConvBnLelu(base_filters, base_filters, kernel_size=3, norm=False, activation=False, bias=False))
        transform_fn = functools.partial(MultiConvBlock, base_filters, int(base_filters * 1.5), base_filters, kernel_size=3, depth=4)
        self.switch = ConfigurableSwitchComputer(base_filters, multiplx_fn, pretransform_fn, transform_fn, trans_count, init_temp=initial_temp,
                                            add_scalable_noise_to_transforms=True, init_scalar=.1)

        self.transformation_counts = trans_count
        self.init_temperature = initial_temp
        self.final_temperature_step = final_temperature_step
        self.heightened_temp_min = heightened_temp_min
        self.heightened_final_step = heightened_final_step
        self.attentions = None
        self.upsample_factor = upsample_factor
        self.backbone_forward = None

    def get_forward_results(self):
        return self.backbone_forward

    def forward(self, x):
        self.backbone_forward = self.backbone_wrapper(F.interpolate(x, scale_factor=2, mode="nearest"))

        x = self.initial_conv(x)

        self.attentions = []
        x, att = self.switch(x, output_attention_weights=True)
        self.attentions.append(att)

        x = self.upconv1(F.interpolate(x, scale_factor=2, mode="nearest"))
        if self.upsample_factor > 2:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.upconv2(x)
        return self.final_conv(self.hr_conv(x)),

    def set_temperature(self, temp):
        self.switch.set_temperature(temp)

    def update_for_step(self, step, experiments_path='.'):
        if self.attentions:
            temp = max(1,
                1 + self.init_temperature * (self.final_temperature_step - step) / self.final_temperature_step)
            if temp == 1 and self.heightened_final_step and step > self.final_temperature_step and \
                    self.heightened_final_step != 1:
                # Once the temperature passes (1) it enters an inverted curve to match the linear curve from above.
                # without this, the attention specificity "spikes" incredibly fast in the last few iterations.
                h_steps_total = self.heightened_final_step - self.final_temperature_step
                h_steps_current = min(step - self.final_temperature_step, h_steps_total)
                # The "gap" will represent the steps that need to be traveled as a linear function.
                h_gap = 1 / self.heightened_temp_min
                temp = h_gap * h_steps_current / h_steps_total
                # Invert temperature to represent reality on this side of the curve
                temp = 1 / temp
            self.set_temperature(temp)
            if step % 50 == 0:
                output_path = os.path.join(experiments_path, "attention_maps", "a%i")
                prefix = "attention_map_%i_%%i.png" % (step,)
                [save_attention_to_image_rgb(output_path % (i,), self.attentions[i], self.transformation_counts, prefix, step) for i in range(len(self.attentions))]

    def get_debug_values(self, step):
        temp = self.switch.switch.temperature
        mean_hists = [compute_attention_specificity(att, 2) for att in self.attentions]
        means = [i[0] for i in mean_hists]
        hists = [i[1].clone().detach().cpu().flatten() for i in mean_hists]
        val = {"switch_temperature": temp}
        for i in range(len(means)):
            val["switch_%i_specificity" % (i,)] = means[i]
            val["switch_%i_histogram" % (i,)] = hists[i]
        return val

