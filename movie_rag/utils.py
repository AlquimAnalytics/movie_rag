"""utils
"""

import json
import os
import chardet
import requests


def get_config(argv):
    """Get the configuration data from the -c or --config process parameter.
    """
    if argv[0] in ['-c', '--config']:
        cfg_fname = argv[1]
        with open(cfg_fname, 'r') as fnum:
            config_text = fnum.read()
            config = json.loads(config_text)
        # Resolve directory-like paths relative to the config file location
        cfg_dir = os.path.dirname(os.path.abspath(cfg_fname))
        for key in ['landing_area', 'staging_area', 'prepared_area']:
            value = config.get(key)
            if isinstance(value, str) and not os.path.isabs(value):
                config[key] = os.path.normpath(os.path.join(cfg_dir, value))

        # Backward-compatibility: allow sql_db_fname as alias for table_store_fname
        if 'table_store_fname' not in config and 'sql_db_fname' in config:
            config['table_store_fname'] = config['sql_db_fname']
    else:
        raise ValueError('No config file specified.')

    return config


def determine_file_encoding(fname, n=None):
    """Determine a file's encoding using chardet.
    """

    with open(fname, 'rb') as fnum:
        if n is None:
            rawdata = fnum.read()
        else:
            rawdata = b''.join([fnum.readline() for _ in range(n)])

    return chardet.detect(rawdata)['encoding']


def is_ollama_running(url="http://localhost:11434"):
    """Determine if Ollama is running and available.
    """

    reply = False
    try:
        response = requests.get(f'{url}/api/tags')
        if response.status_code == 200:
            reply = True
    except requests.exceptions.ConnectionError:
        pass

    return reply
