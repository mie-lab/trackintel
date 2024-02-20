Summarised below are the steps required to run the asv benchmark on a new computer.

1. Clone trackintel and checkout to master. Add the trackintel root to python path

2. Create a new `conda` environment and install the requirements for trackintel. I tested using Ubuntu20 and there were some issues with the requirements. If you are using some other OS, you can skip these are proceed to Step 4

> Some ubuntu specific tweaks:

>1 (a). The `psychopg2` runs into error while doing `pip install -r reqirements.txt`

>> Solution: Use `psychopg2-binary` 
    
> 1(b) If you get an error as shown below:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
spyder 5.1.5 requires pyqt5<5.13, which is not installed.
spyder 5.1.5 requires pyqtwebengine<5.13, which is not installed.

```
>> Solution:
```
pip install --upgrade --user pyqtwebengine==5.12
pip install --upgrade --user pyqt5==5.12
```

3. Install asv using `pip install asv`
4. Change current directory to the root of trackintel. 
5. Get the list of commits in short form (last 7 characters of commit id) which were merge commits in master. For example, if we want the last 3 merge commits, the command is given below: 
```
git log | grep -B 1 'Merge' | grep 'commit' | sed 's/commit //g' | cut -c1-7 | head -n 3 > commits.txt
```
6. Run benchmarks for these commits using:
```
asv run HASHFILE:commits.txt
```
>> If running for the first time, the asv will ask some questions about the computer being run. These are only for documentation purposes and do not affect the numerical value of benchmarks. If all went well, the asv will start setting up the enviroments for each `<benchmark> * <commit>` as shown below:
>> 
>> <img src="https://i.imgur.com/r6pYhNB.png" width="360">
 
>> If you are running for the first time, it is recommended to use the flag `-q` so that each benchmark is run just once so that you know it works (`asv run -q HASHFILE:commits.txt`). Finally if there are no issues, the benchmarks can be rerun without the `-q` option. 
>> If the benchmarks show up as `failed` as shown below:
>> 
>> <img src="https://i.imgur.com/r3vIcgm.png" width="360">

>>> Time for debugging → use the `-e` flag. The error message consists of the entire call stack as shown below:
>>> <img src="https://i.imgur.com/clGNgCJ.png" width="360">


7. If the benchmarks run successfully, more benchmarks can be run. On the same machine, asv remembers the old benchmarks and those can be ignored using the flag using: 
``` 
asv run --skip-existing HASHFILE:commits.txt 
```
> More information on the flags such as running benchmarking between specific versions etc..  can be found at the [asv documentation homepage](https://asv.readthedocs.io/en/stable/commands.html).


8. After all benchmarks have run successfully, we can generate the html files using:
```
asv publish
``` 
> The generated html files are not dynamic and hence cannot be directly opened in a browser. Instead, we can view the files using the following command which starts a localhost server: 

```
asv preview
``` 

9. . Now we need to push the html files to gh-pages branch to host it on the server. The documentation mentions that we can run `asv gh-pages` but it did not work. Instead, we push manually using the commands below:
```
asv gh-pages --no-push --rewrite
```
```
git stash
```
```
git checkout gh-pages 
```
```
git log 
```    
```
git push -f origin gh-pages 
```
>> Sometimes, the first line `asv gh-pages --no-push --rewrite` results in a warning and gets stuck after writing to the gh-pages branch as  shown below. If this happens, it should be terminated with `<Ctrl+C>`. As long as the message shows 100% completed writing to gh-pages, this can be terminated without any problems to the generated stats. 
>> 
>> <img src="https://user-images.githubusercontent.com/9101260/182346530-3cfa83dc-6efe-45ba-91a8-3586369f14fe.png" width="360">



>>`git log` in the step above should show the last commit as "Generated from sources" as shown below: 
>>
>> <img src="https://i.imgur.com/YKZkgAJ.png" width="251">


>>-f in git push is important because we are rewriting the gh-pages branch and it causes some conflicts with the remote. 
 
 
 
10. Finally, we revert back to the original branch (for which we might need to run benchmarks again) and pop the earlier stash.
```
git checkout master
```
```
git stash pop
```

 
 
