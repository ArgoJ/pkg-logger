# package-logger

Small helper package for package-scoped logging with optional `tqdm` progress bar integration.

## Install

```bash
pip install .
```

## Usage

```python
from pkg_logger import PackageLogger, get_package_logger

PackageLogger.setup("my_package", short_name="mp")
logger = get_package_logger("my_package.module")
logger.info("Hello")

with logger.tqdm(range(3), desc="Work") as pbar:
    for _ in pbar:
        logger.info("step")
```
