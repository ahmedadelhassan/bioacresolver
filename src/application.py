#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import uvicorn


if __name__ == "__main__":
    logging.info('****** Starting Service Server ******')
    uvicorn.run('apis:create_app', factory=True, host="0.0.0.0", port=8080)
