from pathlib import Path
from setuptools import setup, find_packages


ROOT = Path(__file__).parent.resolve()


def read_long_description() -> str:
	readme = ROOT / "readme.md"
	if readme.is_file():
		return readme.read_text(encoding="utf-8")
	return "MTinv_OT: Magnetotelluric 1D inversion and 2D forward modeling toolbox."


def read_version() -> str:
	"""从 src/mt1d_inv/__init__.py 读取 __version__，避免直接导入包。"""
	init_path = ROOT / "src" / "mt1d_inv" / "__init__.py"
	about: dict = {}
	if init_path.is_file():
		code = init_path.read_text(encoding="utf-8")
		exec(code, about)  # 只提取 __version__ 等常量
		version = about.get("__version__")
		if version:
			return version
	return "0.0.0"


setup(
	name="mtinv-ot",
	version=read_version(),
	description="Magnetotelluric 1D inversion with OT and 2D FD forward modeling.",
	long_description=read_long_description(),
	long_description_content_type="text/markdown",
	author="lxr, cxz",
	author_email="xinran.liu@zju.edu.cn",
	url="",
	license="",
	python_requires=">=3.8",
	package_dir={"": "src"},
	packages=find_packages("src"),
	include_package_data=True,
	install_requires=[
		"numpy",
		"matplotlib",
		"scikit-image",
		# 用户需自行安装对应版本的 PyTorch
		"torch",
	],
	extras_require={
		"ot": ["geomloss"],
		"dev": [
			"jupyter",
			"pytest",
		],
	},
	classifiers=[
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3 :: Only",
		"Intended Audience :: Science/Research",
		"Topic :: Scientific/Engineering :: Physics",
		"Topic :: Scientific/Engineering :: Information Analysis",
	],
)

