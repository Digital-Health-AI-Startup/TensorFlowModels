from setuptools import setup
with open('README.md') as f:
    readme = f.read()
setup(
    name='ph_tfm',
    version='0.1.0',
    description='Percipio-Health Tensorflow Models',
    long_description=readme,
    author='Percipio-Health',
    author_email='corey@percipiohealth.com',
    url='https://github.com/Digital-Health-AI-Startup/TensorFlowModels',
    packages=['official']
)
