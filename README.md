The report about the experiments is in report.ipynb run 
`jupyter notebook`
to check it

## Training:

`virtualenv venv`

`source venv/bin/activate`

`pip install -r requirements.txt`

and then
(for lab format)

`cd lab`

`python run_lab.py PATH_WITH_IMAGES [args]`

or (for rgb format)

`cd rgb`

`python run_rgb.py PATH_WITH_IMAGES [args]`

the models will be saved in models directory and the details about the training (including sample images)
will be uploaded in real time to 
https://app.neptune.ai/rm360179/image-coloring

## Interference:

To run the previously trained model, run
`python interference.py img_path model_path ouput_path`
in rgb or lab directory

They sample images are much easier to see at neptune website though
