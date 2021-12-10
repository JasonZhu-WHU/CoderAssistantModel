## Model for CoderAssistant

Transform code snippet into text description: code2text

This project is based on [CodeXGlue](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text) and CodeBERT



### How to run the backend model

```
git clone https://github.com/JasonZhu-WHU/CoderAssistantModel.git
```

And then download the model from [here](https://drive.google.com/file/d/1tvI22o3Eybyyrq6bbZ4G89A9WNjHZroj/view?usp=sharing). Unzip the model file and place it at /model/java_model.bin

```
cd code
python app.py
```



### Reference

```
@article{husain2019codesearchnet,
  title={Codesearchnet challenge: Evaluating the state of semantic code search},
  author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  journal={arXiv preprint arXiv:1909.09436},
  year={2019}
}
```

