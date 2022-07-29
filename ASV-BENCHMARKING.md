Summarised below are the steps required to run the asv benchmark on a new computer.

1. Clone trackintel and checkout to master. Add the trackintel root to python path 

>(**For testing purposes only,** clone the abcnishant007/trackintel and checkout to `asv-trackintel` branch). 
*@Ye, we need to edit this line after pull request is approved. *

2. Download the larger geolife data from [this branch](https://github.com/abcnishant007/trackintel/tree/benchmark-files) and place it inside `tests/data/geolife_long_10_MB`. This dataset will not be used by default. The dataset option is hardcoded in the benchmark files as shown [here](https://github.com/abcnishant007/trackintel/blob/7b8c2ee2f12d98b59578cd0519aae6a5240ade4c/benchmarks/preprocessing_benchmarks.py#L6). The hardcoding exists temporarily only to ensure that we can switch to the smaller dataset for asv testing purposes on a new computer. Once the whole setup is complete, we can start using the bigger dataset. The hardcoding can be removed (or retained for future setups) and only the `geolife_long_10_MB` data set should be used for actual benchmarks.
3. Create a new `conda` environment and install the requirements for trackintel. I tested using Ubuntu20 and there were some issues with the requirements. If you are using some other OS, you can skip these are proceed to Step 4

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

4. Install asv using `pip install asv`
5. Change current directory to the root of trackintel. 
6. Get the list of commits in short form (last 7 characters of commit id) which were merge commits in master using the shell command below: 
```git log | grep -B 1 “Merge” | grep “commit” | sed ‘s/commit //g’ | cut -c1-7 | head -n 3 > commits.txt```
7. Run benchmarks for these commits. 
```asv run HASHFILE:commits.txt```
>> If running for the first time, the asv will ask some questions about the computer being run. These are only for documentation purposes and do not affect the numerical value of benchmarks. If all went well, the asv will start setting up the enviroments for each `<benchmark> * <commit>` as shown below:
>>![](https://i.imgur.com/r6pYhNB.png)
>> If you are running for the first time, it is recommended to use the flag `-q` so that each benchmark is run just once so that you know it works (`asv run -q HASHFILE:commits.txt`). Finally if there are no issues, the benchmarks can be rerun without the `-q` option. 
>> If you see benchmarks failed as shown below:
>> ![](https://i.imgur.com/r3vIcgm.png)
>>> Time for debugging → use the `-e` flag. The error message consists of the entire call stack as shown below:
>>>![](https://i.imgur.com/clGNgCJ.png)


8. If the benchmarks run successfully, more benchmarks can be run. On the same machine, asv remembers the old benchmarks and those can be ignored using the flag as ` asv run --skip-existing HASHFILE:commits.txt `
> More information on the flags such as running benchmarking between specific versions etc..  can be found at the [asv documentation homepage](https://asv.readthedocs.io/en/stable/commands.html).


9. After all benchmarks have run successfully, run 
```asv publish``` 
to generate the html files. The generated html files are not dynamic and hence cannot be directly opened in a browser. Instead, 
10. Run 
```asv preview``` 
to view the files through a localhost server
11. Now we need to push the html files to gh-pages branch to host it on the server. The documentation mentions that we can run `asv gh-pages` but it did not work for me. Instead, I push manually as shown below:
```
asv gh-pages --no-push --rewrite
git stash
git checkout gh-pages 
git log 
git push -f origin gh-pages 
```
>>git log should show the last commit as "Generated from sources": 
>>![](https://i.imgur.com/YKZkgAJ.png)

>>-f in git push is important because we are rewriting th gh-pages branch and it causes some conflicts with the remote. 
 
 
 
**For testing purposes only**
 Once this is setup, the `branches` parameter name should be reset to master in the `asv.conf.json file`. Currently this is available only until the pull request is not approved. By default the asv looks for master branch, so throws an error: 
 ```asv.util.ProcessError: Command '/usr/bin/git rev-list --first-parent master' returned non-zero exit status 128```
 *@Ye, we need to edit this line after pull request is approved.*
 
 
