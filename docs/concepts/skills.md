# Skills

## Motivation

Suppose your robot has certain capabilities -- e.g., it can move in certain ways, or play sounds through a speaker. How do you let an LLM agent control these capabilities?

Skills are how you do that: skills get turned into *tools* that agents can call.

Skills are often also defined at a *higher level of abstraction* than the robotic capabilities; e.g., a 'follow human' skill that uses computer vision data to control a robot. In this way, skills can be

* easier for agents to work with and reason about
* and hide or abstract over differences in the underlying hardware.

```python
from dimos.core import Module
from dimos.protocol.skill.skill import skill

class NavigationModule(Module):
    @skill()
    def navigate_to(self, location: str) -> str:
        """Navigate to a named location like 'kitchen'."""
        x, y, theta = self._lookup_location(location)
        self._set_navigation_goal(x, y, theta)
        return f"Navigating to {location}"
```
<!-- Citation: dimos/core/module.py:77 - ModuleBase inherits from SkillContainer -->

Finally, if there's information you want to get to an agent, you need to do that with skills -- more on this shortly.

## What is a skill?

At a high level, skills are wrappers over lower-level robot capabilities. But at a more prosaic level, a skill is just a method on a Module decorated with `@skill` that:

1. **Becomes an agent-callable tool** - The decorator generates an OpenAI-compatible function schema from the method signature and docstring
2. **Executes in background threads** - Skills run concurrently without blocking the agent
3. **Reports state via messages** - Each execution tracks state (pending → running → completed/error)

<!-- Citation: dimos/protocol/skill/skill.py:65-113 - @skill decorator implementation -->

> [!TIP]
> The docstring becomes the tool description LLMs see when choosing skills. Write it for an LLM audience: make it clear, concise, action-oriented.

## Basic usage

### Defining a simple skill

For a method on a `Module` to be discoverable by agents, it has to be decorated with `@skill()` and registered on the agent -- [see the 'equip an agent with skills' tutorial for more details](../tutorials/skill_with_agent/tutorial.md).

```python
from dimos.core import Module
from dimos.protocol.skill.skill import skill

class RobotSkills(Module):
    @skill()
    def speak(self, text: str) -> str:
        """Make the robot speak the given text aloud."""
        self.audio.play_tts(text)
        return f"Said: {text}"

    @rpc
    def set_LlmAgent_register_skills(self, register_skills: RpcCall) -> None:
        """Called by framework when composing with llm_agent().

        This method is discovered by convention during blueprint.build().
        """
        register_skills.set_rpc(self.rpc)
        register_skills(RPCClient(self, self.__class__))
```

> [!NOTE]
> For most scenarios, you can avoid having to repeat the `set_LlmAgent_register_skills` boilerplate by  subclassing `SkillModule` (which is just `Module` plus the `set_LlmAgent_register_skills` method shown above).

### How skills reach agents

When you register a Module with an agent, the agent discovers its `@skill` methods and converts them into *tool schemas* that the LLM understands. Your method signature becomes the tool's parameters; your docstring becomes its description.

See these tutorials for examples:

* [Equip an agent with skills](../tutorials/skill_with_agent/tutorial.md).
* [Build a RoboButler multi-agent system](../tutorials/multi_agent/tutorial.md)

## Updating agents with results from skills

We've seen how skills can be made available to agents as tools they can call. Often, however, we don't just want agents making tool calls -- we also want to relay updates from the tool calls, from the skills, back to the agent.

Some of this behavior already comes as a default: if you decorate the method with `@skill()`, the agent will be notified with the *return value* of the method when the skill finishes (because the default value of the `ret` parameter is `Return.call_agent`).

### Notifying the agent whenever there's updates

But often we want to update the agent not just when the skill is finished, but also whenever there's progress. Think, e.g., of a 'move to certain coordinates' skill, where we might want to stream progress updates continously to the agent.

This can be done by making  the method a generator and setting the `stream` parameter of `@skill` to `Stream.call_agent`:

```python
@skill(stream=Stream.call_agent, reducer=Reducer.string)
def goto(self, x: float, y: float):
    """Move the robot in relative coordinates.
    x is forward, y is left.
    goto(1, 0) will move the robot forward by 1 meter
    """
    pose_to = PoseStamped(
                          # ...
                          )
    yield "moving, please wait..."  # Notifies the agent
    self.navigate_to(pose_to)
    yield "arrived"                 # Notifies the agent again
```
<!-- Citation: dimos/navigation/rosnav.py:305-322 -->

The agent is notified with each `yield`, and can take action if something goes wrong.

### Streaming updates more '*passively*', in the background

That said, we don't always want to update the agent *every time* there's an update. Sometimes we want to just accumulate updates in the background and *only* pass them on to the agent when the agent happens to be notified by other more 'active' skills. For instance, we may want to periodically pass on information from a camera feed without interrupting the agent on every frame.

To do this, use `Stream.passive`:

```python
from dimos.msgs.sensor_msgs import Image

class CameraFeed(Module):
    color_image: Out[Image]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._image_queue = queue.Queue(maxsize=1)

    @rpc
    def start(self):
        super().start()
        self.hardware = # ...
        # Subscribe to hardware stream

    @rpc
    def stop(self) -> None:
        # ...Clean up resources like the hardware stream
        super().stop()


    @skill(stream=Stream.passive, output=Output.image, reducer=Reducer.latest)
    def video_stream(self):
        """Implicit video stream skill"""
        self.hardware.image_stream().subscribe(self._image_queue.put)
        yield from iter(self._image_queue.get, None)
```

Note that the above skill can *also* be called by the agent. If you don't want that -- if you don't want the skill to be available as a tool -- set the `hide_skill` parameter to `True`.

> [!CAUTION]
> **Passive skills alone cannot keep the agent loop alive.** If only passive skills are running, the loop exits immediately. Passive skills need to be paired with other active skills; e.g.:
>
> * Position telemetry (passive) + navigation command (active)
> * Video stream (passive) + `HumanInput` (active)

For more on the `stream` and `ret` parameters, see [the Skills API reference](../api/skills.md).

### Reducers as backpressure buffers for streamed updates

When a skill streams updates, the agent might not process them as fast as they arrive. This is when the `reducer` parameter comes in handy: when updates pile up, the designated reducer is used to combine or aggregate updates. E.g., in the camera feed example above, with `reducer=Reducer.latest`, the agent will only see the latest frame from the camera feed.

> [!NOTE]
> With `Stream.passive`, values accumulate silently until an active skill wakes the agent. With `Stream.call_agent`, whether updates are accumulated depends on whether yields happen faster than the agent processes them.

## Getting information to agents on demand

We've seen how updates from skills can be streamed to agents; in particular, how something like a video stream can be streamed in the background. It's worth noting, though, that another way to give an agent access to information is to give it a skill for getting such information *on demand* (think of coding agents and their search tools).

```python
class GoogleMapsSkillContainer(SkillModule):
    _latest_location: LatLon | None = None
    _client: GoogleMaps

    # ...

    @skill()
    def where_am_i(self, context_radius: int = 200) -> str:
        """This skill returns information about what street/locality/city/etc
        you are in. It also gives you nearby landmarks.

        Example:

            where_am_i(context_radius=200)

        Args:
            context_radius (int): default 200, how many meters to look around
        """
```
<!-- Adapted from dimos/agents2/skills/google_maps_skill_container.py -->

## Best practices

**Return meaningful strings** - `"Navigated to kitchen in 12 seconds"` beats `"ok"` for LLMs.

**Write clear docstrings** - They become tool descriptions. Be specific about what the skill does and what parameters mean.
<!-- Citation: dimos/protocol/skill/schema.py - function_to_schema() extracts docstrings -->

**Handle errors gracefully** - Return contextual error messages for agent recovery, not raw exceptions.

**Monitor long-running skills** - Use `skillspy` to watch skill execution in real-time. Skills are tracked in an execution database showing what's currently running and what has completed—invaluable for debugging navigation or other long operations. See the [skill basics tutorial](../tutorials/skill_basics/tutorial.md) for an example of this.

> [!WARNING]
> **Don't use both `@skill` and `@rpc` decorators on a single method** - The `@skill` wrapper can't be pickled for LCM transport. Use `@skill()` for agent tools, `@rpc` for module-to-module calls.


## See also

- [The Skills API reference](../api/skills.md)

### Related concepts

* [Agents](agent.md) - LLM-based reasoning that invokes skills
* [Modules](modules.md) - The distributed actors that provide skills
* [Blueprints](blueprints.md) - Composing modules and skills into systems
