FROM wildflower/base

RUN apt-get update && apt-get install -y \
        git \
        build-essential \
        && \
    rm -rf /var/lib/apt/lists/*

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN conda install -y click pandas tqdm numpy opencv
RUN pip install logbook


RUN git clone https://github.com/WildflowerSchools/suppose.git && \
    cd suppose && \
    python setup.py install

ENTRYPOINT ["/opt/conda/bin/suppose"]
