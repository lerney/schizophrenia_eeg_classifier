def allocate_data(entity_data, sample_rate, interval_length):
    for entity_info in entity_data:
        num_samples = entity_info['data'].shape[1]
        num_channels = entity_info['data'].shape[0]
        num_intervals = len(entity_info['intervals'])
        samples_per_interval = int(sample_rate * interval_length/1000)
        for i in range(num_intervals):
            start_sample = i * samples_per_interval
            end_sample = start_sample + samples_per_interval
            allocated_data = entity_info['data'][:, start_sample:end_sample]
            entity_info['intervals'][i]['data'] = allocated_data
        entity_info['data'] = []
