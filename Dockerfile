FROM sandai/magi:latest

#Install the other required packages
COPY ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
RUN mkdir -p /workspace

WORKDIR /workspace

#download the cosmos_predict 2 from the git repo and copy all files to the workspace
#remove the previous repo if it exists
RUN rm -rf /workspace/MAGI-1
RUN git clone https://github.com/samiazirar/MAGI-1.git 
RUN mv /workspace/MAGI-1/* /workspace/
RUN mv /workspace/MAGI-1/.git /workspace/
RUN rm -rf /workspace/MAGI-1
RUN git clone https://github.com/SandAI-org/MagiAttention.git
RUN cd MagiAttention && git submodule update --init --recursive
RUN cd MagiAttention && pip install --no-build-isolation .
WORKDIR /workspace
