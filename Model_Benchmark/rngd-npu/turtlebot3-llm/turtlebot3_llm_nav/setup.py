import os
from glob import glob

from setuptools import find_packages, setup

package_name = "turtlebot3_llm_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # 런치 파일
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        # 월드/문서 스니펫
        (os.path.join("share", package_name, "worlds"),
            glob("worlds/*") if glob("worlds/*") else []),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jun",
    maintainer_email="plab2090@gmail.com",
    description="LLM-as-coder closed-loop person-finding navigation for TurtleBot3 (waffle).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "llm_nav_node = turtlebot3_llm_nav.llm_nav_node:main",
            "object_search_node = turtlebot3_llm_nav.object_search_node:main",
        ],
    },
)
