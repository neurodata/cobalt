FROM neurodata/ndreg:py3

RUN pip install tqdm joblib ndexchange
COPY ./scripts/ ./scripts
COPY neurodata.cfg ./scripts/
WORKDIR /run/scripts

# set the python environment
RUN echo "export PYTHONPATH=${PYTHONPATH}:/run/ndmulticore" >> start.sh
# download data to preprocess
#RUN echo "python correct_lavision_bias.py neurodata.cfg" >> start.sh
#RUN echo "python register_brain.py neurodata.cfg" >> start.sh
RUN echo "python download_chunks.py" >> start.sh
RUN echo "python detect_cells.py neurodata.cfg" >> start.sh
RUN chmod +x start.sh

CMD ["./start.sh"]

