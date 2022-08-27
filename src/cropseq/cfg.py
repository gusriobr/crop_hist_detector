import logging
import os
import sys
import tempfile
from pathlib import Path

"""
Default configuration
"""
_conf = {
}

_env_content = {}

logger = logging.getLogger(__name__)


def locate_file_in_pythonpath(file_name):
    """
    Find a file by name acroos the PYTHON_PATH routes
    :param file_name:
    :return:
    """
    for path in sys.path:
        full_path = os.path.join(path, file_name)
        if os.path.exists(full_path):
            return full_path
    return None


def _read_env_file(env_map=None):
    if env_map is None:
        env_map = _env_content

    # intentamos localizar el fichero .env
    # si se ha fijado la variable ENVIRON_PROPERTIES, buscar el fichero indicado, sino, buscar por defect .env
    env_path = None
    env_file = os.environ.get("ENVIRON_PROPERTIES")
    if env_file is not None:
        logger.info("Fichero de propiedades a localizar: {}".format(env_file))
        env_path = locate_file_in_pythonpath(env_file)
        if env_path is None:
            logger.info("Fichero {} no encontrado en syspath, se continua con fichero por defecto .env.")

    if env_path is None:
        logger.info("Fichero de propiedades a localizar: {}".format(env_file))
        env_path = locate_file_in_pythonpath(".env")

    if env_path is None:
        logger.info("Fichero .env no encontrado.")
        return

    logger.info("Localizado fichero de propiedades en [{}].".format(env_path))
    _load_fine_into_dict(env_path, env_map)

    # si encontramos un fichero test.env sobrescribimos las propiedades leídas
    test_env_file = locate_file_in_pythonpath("test.env")
    if test_env_file is not None:
        logger.info("Fichero test.env encontrado: {}".format(test_env_file))
        _load_fine_into_dict(test_env_file, env_map)


def _load_fine_into_dict(file_path, map: dict):
    with open(file_path) as myfile:
        for line in myfile:
            if line.startswith("#") or line.startswith("//"):
                continue
            name, var = line.partition("=")[::2]
            map[name.strip()] = var.strip()


def get(var_name):
    """
    Reads a variable in this order: .env file, system env variable, default configuration
    :param var_name:
    :return:
    """
    if var_name in _env_content:
        return _env_content.get(var_name)
    elif var_name in os.environ:
        return os.environ.get(var_name)
    else:
        return _conf.get(var_name)


def get_base_folder():
    # ../../ hasta la base del proyecto
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def project_file(file_path):
    return os.path.join(get_base_folder(), file_path)


def get_resources_folder():
    folder = get('RESOURCES_FOLDER')
    if not folder:
        # use project base folder as base location
        folder = os.path.join(get_base_folder(), "resources")
    return folder


def get_results_folder():
    folder = get('RESULTS_FOLDER')
    if not folder:
        # use project base folder as base location
        folder = os.path.join(get_base_folder(), "results")
    return folder


def get_tmp_folder():
    folder = get('TMP_FOLDER')
    if not folder:
        folder = tempfile.gettempdir()
    return folder


def _mk_if_not_exists(path):
    """
    Crea directorios padre si no existen
    :param path:
    :return:
    """
    fpath = Path(path)
    parent = fpath.parent.absolute()
    Path(parent).mkdir(parents=True, exist_ok=True)


def results(path):
    file_path = os.path.join(get_results_folder(), path)
    _mk_if_not_exists(file_path)
    return file_path


def resource(path):
    file_path = os.path.join(get_resources_folder(), path)
    _mk_if_not_exists(file_path)
    return file_path


def dlake(path):
    file_path = os.path.join(get_datalake_folder(), path)
    _mk_if_not_exists(file_path)
    return file_path


def tmp(path):
    file_path = os.path.join(get_tmp_folder(), path)
    _mk_if_not_exists(file_path)
    return file_path


##################################################
### Module initialization
##################################################

# Load .env file if exists
_read_env_file()

CONFIGURE_LOG = True


def configLog(level=logging.INFO, filepath="cropseq.log"):
    global CONFIGURE_LOG
    if CONFIGURE_LOG:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)-1s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            handlers=[
                                logging.FileHandler(filepath),
                                logging.StreamHandler(stream=sys.stdout)
                            ])
        CONFIGURE_LOG = False
