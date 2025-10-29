# Jupyter Lab configuration for fruit ripeness project

c = get_config()

# Set default kernel
c.MappingKernelManager.default_kernel_name = 'fruit_ripeness_fixed'

# Automatically select the default kernel for new notebooks
c.NotebookApp.kernel_spec_manager_class = 'jupyter_client.kernelspec.KernelSpecManager'