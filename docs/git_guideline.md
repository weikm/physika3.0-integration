## 分支管理（Workflow）
我们使用基于feature的分支管理规范（[Gitflow Workflow | Atlassian Git Tutorial](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)）。每位开发者开发前从develop分支拉出自己的feature分支，在自己的feature分支进行开发工作，开发测试完成后向develop分支提交merge request，每个版本发版前1周版本管理员会从develop分支拉出该版本的release分支，此后当前版本的bug fix等提交均提交到release分支，同时可基于develop分支开启下一个版本的迭代。当一个版本灰度结束，全量上线后，该版本的release分支合并入master分支。
## 提交信息（Commit Message）
git commit –m "type:description"

type用于说明 commit 的类别，只允许使用下面7个标识:
* feat：     新功能（feature）
* fix：      修补bug
* docs：     文档（documentation）
* style：    格式（不影响代码运行的变动）
* refactor： 重构（即不是新增功能，也不是修改bug的代码变动）
* test：     增加测试
* chore：    构建过程或辅助工具的变动

description用于详细描述commit内容，建议用英文描述以避免字符编码问题。
## 其他准则（Miscellaneous）
* 优先使用git pull --rebase拉取新代码。git pull 本质上是 git fetch + git merge 命令。如果远程分支领先本地分支，并且本地分支也有提交，则应该使用 git pull –rebase，其本质是 git fetch + git rebase，使用变基能保持线性提交记录，方便回溯。
* 每个commit的内容应内聚。请勿将多个修改在一次commit中提交，以保持版本管理的整洁，便于回溯。
* 禁止向保护分支直接提交代码（master、develop、release等）。
