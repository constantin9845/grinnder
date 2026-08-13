import matplotlib.pyplot as plt

def forward_timeline(data, num_parts=32):
    ssd_cpu_ends = data[0]
    cpu_gpu_ends = data[1]
    compute_ends = data[2]

    num_layers = len(ssd_cpu_ends) // num_parts

    # Calculate actual start & duration intervals per layer independently
    def get_layer_blocks(ends_list, is_ssd=False, is_gpu=False, is_cmp=False):
        starts, durations = [], []

        for layer in range(num_layers):
            slice_ends = ends_list[layer * num_parts : (layer + 1) * num_parts]

            # Layer start reference times
            if is_ssd:
                layer_start = 0.0 if layer == 0 else compute_ends[(layer * num_parts) - 1]
            elif is_gpu:
                layer_start = ssd_cpu_ends[layer * num_parts + 31]
            elif is_cmp:
                layer_start = cpu_gpu_ends[layer * num_parts]

            # Partition 0 of this layer starts at layer_start
            layer_starts = [layer_start] + slice_ends[:-1]
            layer_durations = [end - start for start, end in zip(layer_starts, slice_ends)]

            starts.extend(layer_starts)
            durations.extend(layer_durations)

        return starts, durations

    ssd_starts, ssd_durations = get_layer_blocks(ssd_cpu_ends, is_ssd=True)
    cg_starts, cg_durations   = get_layer_blocks(cpu_gpu_ends, is_gpu=True)
    cmp_starts, cmp_durations = get_layer_blocks(compute_ends, is_cmp=True)

    fig, ax = plt.subplots(figsize=(16, 5))

    tracks = [
        ('SSD --> CPU', 2, ssd_starts, ssd_durations, '#4A90E2'),
        ('CPU --> GPU', 1, cg_starts,  cg_durations,  '#F5A623'),
        ('Compute',     0, cmp_starts, cmp_durations, '#7ED321')
    ]

    # Plot bars
    for label, y_pos, starts, durations, single_color in tracks:
        for start, duration in zip(starts, durations):
            ax.barh(
                y=y_pos,
                width=duration,
                left=start,
                height=0.55,
                color=single_color,
                edgecolor='black',
                linewidth=0.3
            )


    # Styling
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([
        'SSD --> CPU\n(loading partitions)',
        'CPU --> GPU\n(Gathering)',
        'Compute'
    ], fontsize=12, fontweight='bold')

    ax.set_xlabel('Time (s)', fontsize=16, fontweight='bold')
    ax.set_title('Forward Pass Timeline', fontsize=18, fontweight='bold', pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    ax.set_xlim(0, max(compute_ends) + 2)

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

def backward_timeline(data, num_parts=32):
    # Unpack stages
    ssd_cpu_ends_parts = data[0] if len(data) > 0 and data[0] else []
    ssd_cpu_ends_grad  = data[1] if len(data) > 1 and data[1] else []
    cpu_gpu_ends_parts = data[2] if len(data) > 2 and data[2] else []
    cpu_gpu_ends_grad  = data[3] if len(data) > 3 and data[3] else []
    compute_ends       = data[4] if len(data) > 4 and data[4] else []
    gds_ends           = data[5] if len(data) > 5 and data[5] else []

    # Calculate intervals purely for active time; leading idle time remains white
    def get_layer_blocks(ends_list):
        if not ends_list:
            return [], []

        num_layers = len(ends_list) // num_parts
        starts, durations = [], []

        for layer in range(num_layers):
            slice_ends = ends_list[layer * num_parts : (layer + 1) * num_parts]

            # Calculate average duration per partition in this layer
            if len(slice_ends) > 1:
                avg_part_duration = (slice_ends[-1] - slice_ends[0]) / (len(slice_ends) - 1)
            else:
                avg_part_duration = 0.1

            # Estimate start of partition 0 using avg_part_duration
            p0_start = max(0.0, slice_ends[0] - avg_part_duration)

            for i in range(len(slice_ends)):
                start = p0_start if i == 0 else slice_ends[i - 1]
                end = slice_ends[i]

                starts.append(start)
                durations.append(max(0.0, end - start))

        return starts, durations

    s0_starts, s0_durations = get_layer_blocks(ssd_cpu_ends_parts)
    s1_starts, s1_durations = get_layer_blocks(ssd_cpu_ends_grad)
    s2_starts, s2_durations = get_layer_blocks(cpu_gpu_ends_parts)
    s3_starts, s3_durations = get_layer_blocks(cpu_gpu_ends_grad)
    s4_starts, s4_durations = get_layer_blocks(compute_ends)
    s5_starts, s5_durations = get_layer_blocks(gds_ends)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Reordered tracks (top to bottom: SSD Parts -> CPU->GPU Parts -> SSD Grad -> CPU->GPU Grad -> Compute -> GDS)
    tracks = [
        ('SSD --> CPU (Parts)', 5, s0_starts, s0_durations, '#4A90E2'),
        ('CPU --> GPU (Parts)', 4, s2_starts, s2_durations, '#F5A623'),
        ('SSD --> CPU (Grad)',  3, s1_starts, s1_durations, '#BD10E0'),
        ('CPU --> GPU (Grad)',  2, s3_starts, s3_durations, '#9013FE'),
        ('Backward Compute',    1, s4_starts, s4_durations, '#7ED321'),
        ('GDS Stream',          0, s5_starts, s5_durations, '#E67E22'),
    ]

    # Plot bars (active periods only)
    for label, y_pos, starts, durations, single_color in tracks:
        for start, duration in zip(starts, durations):
            if duration > 0:
                ax.barh(
                    y=y_pos,
                    width=duration,
                    left=start,
                    height=0.55,
                    color=single_color,
                    edgecolor='black',
                    linewidth=0.3
                )

    # Styling with updated labels matching the exact order requested
    ax.set_yticks([5, 4, 3, 2, 1, 0])
    ax.set_yticklabels([
        'SSD --> CPU\n(loading partitions)',
        'CPU --> GPU\n(Gathering partitions)',
        'SSD --> CPU\n(loading gradients)',
        'CPU --> GPU\n(Gathering gradients)',
        'Backward Compute',
        'GDS Stream'
    ], fontsize=11, fontweight='bold')

    ax.set_xlabel('Time (s)', fontsize=16, fontweight='bold')
    ax.set_title('Backward Pass Timeline', fontsize=18, fontweight='bold', pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    all_lists = [ssd_cpu_ends_parts, ssd_cpu_ends_grad, cpu_gpu_ends_parts, cpu_gpu_ends_grad, compute_ends, gds_ends]
    valid_maxes = [max(l) for l in all_lists if len(l) > 0]
    max_x = max(valid_maxes) if valid_maxes else 10
    ax.set_xlim(82, max_x + 2)

    plt.tight_layout()
    plt.show()

forward_partition_load_CPU_timesteps = [
    0.58,
    1.12,
    1.65,
    2.17,
    2.71,
    3.23,
    3.76,
    4.29,
    4.83,
    5.33,
    5.87,
    6.4,
    6.94,
    7.48,
    7.98,
    8.52,
    9.03,
    9.56,
    10.1,
    10.61,
    11.13,
    11.63,
    12.16,
    12.66,
    13.17,
    13.71,
    14.23,
    14.75,
    15.27,
    15.78,
    16.29,
    16.8,
    46.07,
    46.25,
    46.45,
    46.64,
    46.83,
    47.03,
    47.22,
    47.47,
    48.56,
    48.75,
    49.41,
    49.6,
    49.98,
    50.17,
    50.42,
    50.62,
    50.84,
    51.03,
    51.24,
    51.43,
    51.62,
    51.81,
    52.0,
    52.19,
    52.38,
    52.57,
    52.76,
    52.95,
    53.13,
    53.32,
    53.5,
    53.69,
    67.02,
    67.21,
    67.41,
    67.61,
    67.81,
    68.0,
    68.2,
    68.39,
    68.59,
    68.77,
    68.96,
    69.16,
    69.35,
    69.55,
    69.73,
    69.93,
    70.11,
    70.31,
    70.5,
    70.69,
    70.88,
    71.06,
    71.25,
    71.44,
    71.62,
    71.82,
    72.0,
    72.18,
    72.37,
    72.56,
    72.74,
    72.92,
]

forward_partition_load_GPU_timesteps = [
    16.81,
    17.43,
    18.73,
    19.27,
    19.95,
    20.99,
    21.96,
    22.5,
    23.31,
    24.29,
    25.05,
    25.7,
    26.53,
    27.09,
    27.87,
    28.68,
    29.32,
    30.63,
    32.03,
    33.06,
    33.95,
    34.78,
    35.8,
    36.58,
    37.4,
    38.19,
    39.36,
    40.09,
    41.09,
    41.88,
    43.09,
    44.23,
    53.69,
    53.82,
    54.19,
    54.57,
    54.96,
    55.36,
    55.78,
    56.17,
    56.58,
    57.0,
    57.42,
    57.8,
    58.21,
    58.61,
    59.02,
    59.43,
    59.82,
    60.26,
    60.68,
    61.11,
    61.55,
    61.95,
    62.37,
    62.76,
    63.17,
    63.56,
    63.97,
    64.39,
    64.79,
    65.18,
    65.59,
    66.0,
    72.92,
    73.08,
    73.2,
    73.32,
    73.42,
    73.6,
    73.84,
    73.98,
    74.18,
    74.43,
    74.61,
    74.76,
    74.95,
    75.07,
    75.26,
    75.46,
    75.63,
    75.96,
    76.35,
    76.6,
    76.81,
    77.03,
    77.28,
    77.5,
    77.72,
    77.91,
    78.21,
    78.36,
    78.59,
    78.79,
    79.12,
    79.4,
]

forward_compute_timesteps = [
    17.98,
    18.91,
    19.46,
    20.45,
    21.15,
    22.1,
    22.64,
    23.47,
    24.31,
    25.2,
    25.85,
    26.69,
    27.26,
    28.01,
    28.83,
    29.34,
    30.8,
    32.19,
    33.2,
    34.08,
    34.91,
    35.93,
    36.72,
    37.53,
    38.33,
    39.49,
    40.22,
    41.23,
    42.02,
    43.22,
    44.36,
    45.45,
    53.87,
    54.25,
    54.63,
    55.02,
    55.42,
    55.84,
    56.23,
    56.64,
    57.06,
    57.47,
    57.86,
    58.27,
    58.68,
    59.08,
    59.5,
    59.88,
    60.32,
    60.75,
    61.19,
    61.62,
    62.01,
    62.43,
    62.82,
    63.23,
    63.62,
    64.03,
    64.45,
    64.85,
    65.24,
    65.66,
    66.06,
    66.42,
    73.13,
    73.26,
    73.37,
    73.47,
    73.65,
    73.89,
    74.03,
    74.24,
    74.48,
    74.66,
    74.82,
    75.01,
    75.12,
    75.31,
    75.52,
    75.68,
    76.01,
    76.4,
    76.65,
    76.86,
    77.08,
    77.33,
    77.55,
    77.77,
    77.96,
    78.25,
    78.42,
    78.65,
    78.84,
    79.17,
    79.45,
    79.71,
]

loss_partition_load_CPU_timesteps = [
    79.76,
    79.78,
    80.02,
    80.07,
    80.12,
    80.16,
    80.2,
    80.25,
    80.31,
    80.35,
    80.39,
    80.45,
    80.49,
    80.54,
    80.58,
    80.62,
    80.66,
    80.7,
    80.74,
    80.79,
    80.83,
    80.87,
    80.91,
    80.95,
    81.0,
    81.04,
    81.08,
    81.12,
    81.17,
    81.21,
    81.25,
    81.29,
]

loss_partition_load_GPU_timesteps = [
    79.76,
    79.78,
    80.02,
    80.07,
    80.12,
    80.16,
    80.2,
    80.25,
    80.31,
    80.35,
    80.39,
    80.45,
    80.49,
    80.54,
    80.58,
    80.62,
    80.66,
    80.7,
    80.74,
    80.79,
    80.83,
    80.87,
    80.91,
    80.95,
    81.0,
    81.04,
    81.08,
    81.12,
    81.17,
    81.21,
    81.25,
    81.29,
]

backward_partition_load_CPU_timesteps = [
]

backward_partition_load_GPU_timesteps = [
    81.34,
    81.46,
    84.52,
    85.24,
    85.83,
    86.39,
    86.84,
    87.6,
    88.49,
    89.09,
    90.07,
    91.1,
    92.06,
    92.78,
    93.67,
    94.2,
    95.12,
    95.91,
    96.67,
    98.08,
    99.8,
    100.86,
    101.86,
    102.86,
    104.02,
    105.07,
    106.03,
    106.95,
    108.42,
    109.22,
    110.45,
    111.39,
    123.46,
    123.72,
    126.69,
    127.32,
    127.88,
    128.45,
    128.9,
    129.77,
    130.77,
    131.51,
    132.46,
    133.44,
    134.27,
    135.06,
    135.93,
    136.42,
    137.31,
    138.05,
    138.8,
    140.21,
    141.89,
    142.92,
    143.91,
    144.96,
    146.07,
    147.0,
    147.89,
    148.87,
    150.34,
    151.19,
    152.38,
    153.33,
    164.87,
    165.67,
    166.41,
    167.11,
    167.85,
    168.98,
    170.02,
    170.7,
    171.72,
    172.9,
    173.77,
    174.57,
    175.51,
    176.12,
    177.01,
    177.97,
    178.72,
    180.42,
    182.03,
    183.22,
    184.26,
    185.28,
    186.5,
    187.45,
    188.45,
    189.41,
    190.81,
    191.7,
    192.9,
    193.89,
    195.33,
    196.64,
]

backward_gradient_load_CPU_timesteps = [
    123.67,
    123.95,
    126.92,
    127.54,
    128.12,
    128.69,
    129.14,
    129.99,
    130.99,
    131.71,
    132.71,
    133.67,
    134.47,
    135.32,
    136.16,
    136.62,
    137.59,
    138.27,
    139.04,
    140.45,
    142.14,
    143.11,
    144.13,
    145.21,
    146.29,
    147.21,
    148.13,
    149.09,
    150.56,
    151.42,
    152.6,
    153.55,
    165.09,
    165.84,
    166.6,
    167.3,
    168.07,
    169.14,
    170.18,
    170.89,
    171.89,
    173.06,
    173.95,
    174.75,
    175.68,
    176.29,
    177.18,
    178.15,
    178.89,
    180.64,
    182.19,
    183.37,
    184.43,
    185.45,
    186.66,
    187.62,
    188.61,
    189.59,
    190.97,
    191.87,
    193.06,
    194.06,
    195.48,
    196.8,
]

backward_gradient_load_GPU_timesteps = [
    123.67,
    123.95,
    126.92,
    127.54,
    128.12,
    128.69,
    129.14,
    129.99,
    130.99,
    131.71,
    132.71,
    133.68,
    134.47,
    135.32,
    136.16,
    136.62,
    137.59,
    138.27,
    139.04,
    140.46,
    142.14,
    143.13,
    144.14,
    145.21,
    146.29,
    147.21,
    148.13,
    149.09,
    150.56,
    151.42,
    152.6,
    153.55,
    165.09,
    165.84,
    166.6,
    167.3,
    168.31,
    169.14,
    170.18,
    170.89,
    171.89,
    173.06,
    173.95,
    174.75,
    175.68,
    176.29,
    177.18,
    178.15,
    178.89,
    180.64,
    182.19,
    183.37,
    184.43,
    185.45,
    186.66,
    187.62,
    188.61,
    189.59,
    190.97,
    191.87,
    193.06,
    194.07,
    195.48,
    196.81,
]

backward_compute_timesteps = [
    84.52,
    85.24,
    85.82,
    86.39,
    86.84,
    87.6,
    88.49,
    89.09,
    90.07,
    91.1,
    92.06,
    92.78,
    93.67,
    94.19,
    95.12,
    95.87,
    96.64,
    98.08,
    99.8,
    100.86,
    101.86,
    102.86,
    104.02,
    105.07,
    106.03,
    106.95,
    108.42,
    109.22,
    110.45,
    111.39,
    112.99,
    114.35,
    126.69,
    127.32,
    127.88,
    128.45,
    128.9,
    129.77,
    130.77,
    131.51,
    132.46,
    133.44,
    134.27,
    135.05,
    135.93,
    136.42,
    137.31,
    138.05,
    138.8,
    140.21,
    141.89,
    142.92,
    143.91,
    144.96,
    146.07,
    147.0,
    147.89,
    148.87,
    150.34,
    151.19,
    152.38,
    153.33,
    154.86,
    156.25,
    165.67,
    166.41,
    167.11,
    167.66,
    168.98,
    170.01,
    170.7,
    171.72,
    172.9,
    173.77,
    174.57,
    175.51,
    176.12,
    177.01,
    177.97,
    178.72,
    180.42,
    182.03,
    183.22,
    184.26,
    185.28,
    186.5,
    187.45,
    188.45,
    189.41,
    190.81,
    191.7,
    192.9,
    193.89,
    195.33,
    196.64,
    198.01,
]

backward_direct_load_timesteps = [
]

forward_timeline([forward_partition_load_CPU_timesteps,forward_partition_load_GPU_timesteps,forward_compute_timesteps])

backward_data = [
    backward_partition_load_CPU_timesteps,
    backward_gradient_load_CPU_timesteps,
    backward_partition_load_GPU_timesteps,
    backward_gradient_load_GPU_timesteps,
    backward_compute_timesteps,
    backward_direct_load_timesteps
]


backward_timeline(backward_data)