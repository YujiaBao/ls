from setuptools import setup, find_packages

setup(
    name = 'ls',
    version = '1.0',
    author = 'Yujia Bao',
    author_email = 'bao@yujia.io',
    packages = find_packages() + ['ls/configs'],
    include_package_data = True,
    description = 'Learning to Split for Automatic Bias Detection',
    url = 'https://github.com/yujiabao/ls',
    install_requires = [
        'numpy>=1.23.2',
        'PyYAML>=6.0',
        'rich>=12.5.1',
        'scikit-learn>=1.1.2',
        'scipy>=1.9.1',
        'torch>=1.12.1',
        'torchaudio>=0.12.1',
        'torchvision>=0.13.1',
        'torchtext>=0.13.0',
        'tqdm>=4.64.0',
        'transformers>=4.21.2',
    ]
)
