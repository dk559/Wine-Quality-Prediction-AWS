FROM centos:7

RUN yum -y update && yum -y install python3 python3-dev python3-pip python3-virtualenv \
	java-1.8.0-openjdk wget

RUN python -V
RUN python3 -V

ENV PYSPARK_DRIVER_PYTHON python3
ENV PYSPARK_PYTHON python3

RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas

RUN cd /opt && wget https://apache.osuosl.org/spark/spark-3.4.3/spark-3.4.3-bin-hadoop3.tgz  && tar -xzf spark-3.4.3-bin-hadoop3.tgz && rm spark-3.4.3-bin-hadoop3.tgz


RUN ln -s /opt/spark-3.4.3-bin-hadoop3 /opt/spark
RUN (echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc && echo 'export PATH=$SPARK_HOME/bin:$PATH' >> ~/.bashrc && echo 'export PYSPARK_PYTHON=python3' >> ~/.bashrc)

RUN mkdir /wineapp
COPY WineQualityPrediction.py /wineapp/ 

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN /bin/bash -c "source ~/.bashrc"
RUN /bin/sh -c "source ~/.bashrc"

WORKDIR /wineapp

ENTRYPOINT ["/opt/spark/bin/spark-submit", "--packages", "org.apache.hadoop:hadoop-aws:2.7.7", "WineQualityPrediction.py"]