def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"


def calculate_intervals(entity_data, interval_length, sample_rate):
    for entity_info in entity_data:
        entity_info['intervals'] = []
        total_duration = entity_info['data'].shape[1] / sample_rate
        num_intervals = int(total_duration / (interval_length/1000))
        for i in range(num_intervals):
            start_time = float(i * interval_length/1000)
            end_time = float((i + 1) * interval_length/1000)
            interval = {
                'startTime': format_time(start_time),
                'endTime': format_time(end_time),
                'bands': {}
            }
            entity_info['intervals'].append(interval)


