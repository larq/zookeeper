## Contributing

If you want to contribute to Zookeeper and make it better, your help is very welcome. Contributing is also a great way to learn more about social coding on GitHub, new technologies and their ecosystems and how to make constructive, helpful bug reports, feature requests and the noblest of all contributions: a good, clean pull request.

### How to make a clean pull request

- Create a personal fork of the project on GitHub.
- Clone the fork on your local machine via the commandline instructions below. Your remote repo on GitHub is called `origin`.
  - git clone https://github.com/your-username/zookeeper.git
- Add the original repository as a remote called `upstream`.
- If you created your fork a while ago be sure to pull upstream changes into your local repository.
- Create a new branch to work on! Branch from `master`.
- Implement/fix your feature, comment your code.
- Follow the code style of the project, including indentation.
- Use the following command to test your changes:
  - `pytest .`
- Write or adapt tests as needed.
- Add or change the documentation as needed.
- Push your branch to your fork on Github, the remote `origin`.
- From your fork open a pull request in the correct branch. Target the project's `master` branch.
- Wait for approval.
- Once the pull request is approved and merged you can pull the changes from `upstream` to your local repo and delete
your extra branch(es).

And last but not least: Your commit message should describe what the commit, when applied, does to the code – not what you did to the code.
