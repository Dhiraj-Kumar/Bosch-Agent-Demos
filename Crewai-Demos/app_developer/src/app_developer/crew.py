from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class AppDeveloper():
    """AppDeveloper crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def frontend_engineer(self) -> Agent:
        return Agent(
            # type: ignore[index]
            config=self.agents_config['frontend_engineer'],
            verbose=True
        )

    @agent
    def backend_engineer(self) -> Agent:
        return Agent(
            # type: ignore[index]
            config=self.agents_config['backend_engineer'],
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def build_frontend(self) -> Task:
        return Task(
            config=self.tasks_config['build_frontend'],  # type: ignore[index]
            output_file="output/frontend.html",
        )

    @task
    def build_backend(self) -> Task:
        return Task(
            config=self.tasks_config['build_backend'],  # type: ignore[index]
            # This allows the task to execute code in the output (e.g. to validate code structure)
            allow_code_execution=True,
            code_execution_mode="safe",
            max_execution_time=500,
            max_retry_limit=3,
            output_file="output/backend.js"
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AppDeveloper crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
