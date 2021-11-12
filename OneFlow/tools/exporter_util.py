import json
import requests


metric_info_dict = {
  'DCGM_FI_DEV_FB_USED': (9440, "kj", 'GPU memory used (in MiB).'),
  'DCGM_FI_DEV_POWER_USAGE': (9440, "kj", 'GPU power usage (in W).'),
  'DCGM_FI_PROF_GR_ENGINE_ACTIVE': (9440, "kj", 'GPU utilization (in %).'),
  'node_memory_Committed_AS_bytes': (9100, "kj", 'Node Memory information field Committed (in bytes)'),
}


def get_node_metric(metric, node_ip, start, end, step=3, url='http://10.105.1.136:9090/api/v1/query_range'):
    assert metric in metric_info_dict

    def build_params():
        instance = f'{node_ip}:{metric_info_dict[metric][0]}'
        query = metric + '{instance="' + instance + '",job="' + metric_info_dict[metric][1] + '"}'
        params = (
            ('query', query),
            ('start', f'{start}'),
            ('end', f'{end}'),
            ('step', f'{step}'),
        )
        return params

    def metric_format(metric):
        '''
        convert values to float list
        '''
        values = metric['values']
        if len(values) == 0:
            return
        metric['metric']['start'] = values[0][0]
        metric['metric']['end'] = values[-1][0]
        values = [float(v[1]) for v in values]
        metric['metric']['values'] = values
        return metric['metric']

    metric_dict = requests.get(url, params=build_params()).json()
    assert 'data' in metric_dict
    assert 'result' in metric_dict['data']

    metrics = metric_dict['data']['result']
    formated_metrics = [metric_format(metric) for metric in metrics]
    if not formated_metrics:
        formated_metrics.append({
            '__name__': metric,
            'instance': f'{node_ip}',
            'start': start,
            'end': end,
            'values': 'NA',
        })
    return formated_metrics


def get_metrics_of_node(node_ip, start, end, step=3, url='http://10.105.1.136:9090/api/v1/query_range'):
    return {metric: get_node_metric(metric, node_ip, start, end, step, url) for metric in metric_info_dict}


if __name__ == "__main__":
    start = 1636434583.4633102
    end = 1636436677.8328712
    node_metrics = get_metrics_of_node('10.105.0.32', start, end)

    #print(json.dumps(node_metrics, indent=4, sort_keys=True))
    #print(json.dumps(node_metrics, indent=4, sort_keys=True))
    #print(json.dumps(node_metrics['DCGM_FI_DEV_FB_USED'], indent=4, sort_keys=True))
    for key in metric_info_dict.keys():
        v = node_metrics[key]
        if len(v) > 1:
            # take gpu0's metric
            for gpu_metric in v:
                if gpu_metric['gpu'] == '0':
                    values = gpu_metric['values']
                    break
        elif len(v) == 1:
            values = v[0]['values']
        print(key, values)
