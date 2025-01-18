import logging
import sys
import time
from pathlib import Path


def create_logger(logger_output_dir, net_name, dataset_name, phase):
    root_output_dir = Path(logger_output_dir)
    # <---------log_dir config---------->
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    dataset = dataset_name
    final_output_dir = root_output_dir / dataset / net_name
    if not final_output_dir.exists():
        print('=> creating {}'.format(final_output_dir))
        final_output_dir.mkdir(parents=True, exist_ok=True)

    # <---------log config--------->
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(net_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    logging.basicConfig(filename=str(final_log_file),
                        format='%(asctime)-15s %(message)s',
                        level=logging.INFO)
    logger = logging.getLogger()
    console = logging.StreamHandler(stream=sys.stdout)
    logging.getLogger().addHandler(console)

    return logger

