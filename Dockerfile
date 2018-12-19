FROM neurodata/ndreg:py3

#ARG config=neurodata.cfg
#COPY ./scripts/ ./scripts

#WORKDIR /run/scripts
#RUN mkdir -p process_folder && python bloby_exec.py $config

# run the rest of the scripts
#RUN python scripts/correct_lavision_bias.py -h 
#RUN echo "running bias correction" && \
#    python correct_lavision_bias.py --config $config
RUN pip install tqdm joblib
COPY ./scripts/ ./scripts
WORKDIR /run/scripts

# download the chunks
# set the python environment
RUN echo "export PYTHONPATH=${PYTHONPATH}:/run/ndmulticore" >> start.sh
RUN echo "python cobalt.py" >> start.sh
RUN chmod +x start.sh

CMD ["./start.sh"]

