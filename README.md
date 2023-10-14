# Genre-Classification

### DVC 
[DVC documentation](https://dvc.org/doc/start)

Install dvc and dvc-s3
```shell
pipenv install dvc dvc-s3 -d
```

Initialize DVC in your local project directory
```shell
dvc init
```
This will create a .dvc directory that will store all the DVC related files.

Add Data to DVC
```shell
dvc add data/.
```
This will create a .dvc file that tracks your data.

To add a remote go to the repository in DagsHub and select the dvc option and follow the instructions.

Commit Changes
```shell
git add .  git commit -m "message"
```
```shell
git commit -m "message"
```

Push to DagsHub
```shell
git push origin master
```
Push Data to DVC Remote
```shell
dvc push -r origin
```

Pull Changes
```shell
git pull origin master
```
```shell
dvc pull
```
