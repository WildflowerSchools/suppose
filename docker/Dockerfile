FROM wildflower/base

MAINTAINER lue.her@gmail.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/WildflowerSchools/suppose.git && \
    cd suppose && \
    python setup.py install

ENTRYPOINT ["/opt/conda/bin/suppose"]
