FROM wildflower/base

MAINTAINER lue.her@gmail.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /data

ENV TF_POSE_GIT_REVISION 7353c148d3d9b2609d7592be47e9844d6ea4f46f
RUN git clone https://github.com/ildoonet/tf-pose-estimation.git && \
    cd tf-pose-estimation && \
    git checkout $TF_POSE_GIT_REVISION && \
    python setup.py install && \
    cd .. && \
    rm -rf tf-pose-estimation

ENV SUPPOSE_GIT_REVISION 321e20672d209643f48a718836575956a60d4f00
RUN git clone https://github.com/WildflowerSchools/suppose.git && \
    cd suppose && \
    git checkout $SUPPOSE_GIT_REVISION && \
    python setup.py install && \
    cd .. && \
    rm -rf suppose

ENTRYPOINT ["/opt/conda/bin/suppose"]
