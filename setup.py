import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tabecmo",
    version="0.0.1",
    author="Isaac Sears",
    author_email="isaac.j.sears@gmail.com",
    description="Tabnet ECMO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["tabecmo"],
    url="https://github.com/isears/tabecmo",
    project_urls={
        "Bug Tracker": "https://github.com/isears/tabecmo/issues",
    },
)
