FROM wildflower/base

MAINTAINER lue.her@gmail.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ildoonet/tf-pose-estimation.git && \
    cd tf-pose-estimation && \
    git checkout 7353c148d3d9b2609d7592be47e9844d6ea4f46f && \
    python setup.py install

ENV SUPPOSE_GIT_REVISION 5747757ca6be0fef1464299c71863d371fb45c1c

RUN git clone https://github.com/WildflowerSchools/suppose.git && \
    cd suppose && \
    git checkout $SUPPOSE_GIT_REVISION && \
    python setup.py install

ENTRYPOINT ["/opt/conda/bin/suppose"]
