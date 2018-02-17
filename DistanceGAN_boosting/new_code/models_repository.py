import os
from collections import Counter
import torch


def _get_last_version(path):
    if not os.path.isdir(path):
        return None
    model_files = [os.path.join(path, x) for x in os.listdir(path)]
    files_versions = [f.split('-')[-1] for f in model_files]
    versions_counter = Counter(files_versions)
    versions_with_all_models = [k for k, v in versions_counter.items() if v == 4]
    return max(versions_with_all_models)


class ModelsRepository:
    def __init__(self, models_path):
        self.models_path = models_path

    def _get_G_path(self, G1, path=None):
        return os.path.join(path if path else self.models_path, 'G1' if G1 else 'G2')

    def save_model(self, gen_a, gen_b, dis_a, dis_b, version, G1=True):
        path = self._get_G_path(G1)
        print('Saving {0} version {1} to path {2}'.format('G1' if G1 else 'G2', version, path))
        os.makedirs(path, exist_ok=True)
        torch.save(gen_a, os.path.join(path, 'gen_A-' + version))
        torch.save(gen_b, os.path.join(path, 'gen_B-' + version))
        torch.save(dis_a, os.path.join(path, 'dis_A-' + version))
        torch.save(dis_b, os.path.join(path, 'dis_B-' + version))

    def has_models(self, G1=True):
        path = self._get_G_path(G1)
        return _get_last_version(path) is not None

    def get_models(self, G1=True, path=None, wanted_version=None):
        path = self._get_G_path(G1, path)
        last_version = _get_last_version(path)
        if last_version is None:
            return None
        if wanted_version is not None:
            last_version = wanted_version
        print('Loading {0} version {1} from path {2}'.format('G1' if G1 else 'G2', last_version, path))
        gen_a = torch.load(os.path.join(path, 'gen_A-' + str(last_version)))
        gen_b = torch.load(os.path.join(path, 'gen_B-' + str(last_version)))
        dis_a = torch.load(os.path.join(path, 'dis_A-' + str(last_version)))
        dis_b = torch.load(os.path.join(path, 'dis_B-' + str(last_version)))
        return gen_a, gen_b, dis_a, dis_b, int(last_version)
