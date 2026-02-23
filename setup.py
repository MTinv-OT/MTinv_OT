import re
from pathlib import Path
from setuptools import setup, find_packages


ROOT = Path(__file__).parent.resolve()


def read_long_description() -> str:
	for name in ("README.md", "readme.md"):
		readme = ROOT / name
		if readme.is_file():
			return readme.read_text(encoding="utf-8")
	return "MTinv_OT: Magnetotelluric 1D inversion and 2D forward modeling toolbox."


def read_version() -> str:
	"""Read __version__ from src/mt1d_inv/__init__.py without importing (avoids exec side effects)."""
	init_path = ROOT / "src" / "mt1d_inv" / "__init__.py"
	if init_path.is_file():
		text = init_path.read_text(encoding="utf-8")
		m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
		if m:
			return m.group(1)
	return "0.0.0"


setup(
	name="mtinv-ot",
	version=read_version(),
	description="Magnetotelluric 1D inversion with OT and 2D FD forward modeling.",
	long_description=read_long_description(),
	long_description_content_type="text/markdown",
	author="lxr, cxz，by",
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
		# User should install PyTorch version manually
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

