import yaml


class ADNetConf:
    conf = None

    @staticmethod
    def get(cfg=None):
        if cfg is not None:
            ADNetConf.conf = ADNetConf(cfg)
        return ADNetConf.conf

    @staticmethod
    def g():
        return ADNetConf.get()

    def __init__(self, path):
        if path:
            with open(path, 'r') as fp:
                self.conf = yaml.load(fp)

    def __getitem__(self, key):
        return self.conf[key]

if __name__ == '__main__':
    cf = ADNetConf.get('conf/large.yaml')
    pass
