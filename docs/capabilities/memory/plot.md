
# color cycle

fourteen curves with no explicit color. the auto-cycle yields colors in this
order: the curated six (`color.blue`, `color.red`, `color.yellow`,
`color.teal`, `color.purple`, `color.orange`), then six gap-subdivided
extensions (`color.green`, `color.magenta`, `color.indigo`, `color.cyan`,
`color.vermilion`, `color.amber`), and after that an unbounded golden-angle
hue walk anchored on the average lightness/saturation of the named palette.
curves 13 and 14 cross into that fallback.

```python session=plot output=none
import math
import random

from dimos.memory2.vis.plot.elements import Series
from dimos.memory2.vis.plot.plot import Plot

rng = random.Random(42)
xs = [i * 0.1 for i in range(120)]

color_check = Plot()
for i in range(14):
    phase = rng.uniform(0, 2 * math.pi)
    freq = rng.uniform(0.5, 1.8)
    amp = rng.uniform(0.6, 1.4)
    offset = i * 0.5  # vertical separation so curves don't overlap
    ys = [amp * math.sin(freq * x + phase) + offset for x in xs]
    color_check.add(Series(ts=xs, values=ys, label=f"curve {i + 1}"))

color_check.to_svg("assets/plot_colors.svg")
```








![output](assets/plot_colors.svg)

named colors can also be used explicitly. when you pin a series to one of
the named colors, the auto-cycle excludes it for the remaining series, so
you never end up with two lines that share a color by accident.

```python session=plot output=none
from dimos.memory2.vis import color

p = Plot()
p.add(Series(ts=xs, values=[math.sin(x) for x in xs]))                       # auto → blue
p.add(Series(ts=xs, values=[math.cos(x) for x in xs], color=color.red))      # explicit red
p.add(Series(ts=xs, values=[math.sin(2 * x) for x in xs]))                   # auto → yellow (red is excluded)
p.to_svg("assets/plot_named.svg")
```








![output](assets/plot_named.svg)

# speed plot

```python output=none
from dimos.memory2.store.sqlite import SqliteStore
from dimos.memory2.transform import smooth, speed, throttle
from dimos.memory2.vis import color
from dimos.memory2.vis.plot.elements import Series
from dimos.memory2.vis.plot.plot import Plot
from dimos.utils.data import get_data

store = SqliteStore(path=get_data("go2_bigoffice.db"))
images = store.streams.color_image

plot = Plot()
plot.add(
    images.transform(speed()).transform(smooth(30)),
    label="speed (m/s)",
    opacity=0.5
)

plot.add(
    images.transform(throttle(0.5)).map_data(lambda obs: obs.data.brightness).transform(smooth(10)),
    label="brightness",
    color=color.blue,
)

plot.add(
    images.transform(throttle(0.5)).scan_data(images.first().ts, lambda state, obs: [state, obs.ts - state]),
    label="time",
    axis="time",
    opacity=0.33
)

from dimos.models.embedding.clip import CLIPModel
clip = CLIPModel()
search_vector = clip.embed_text("door")

plot.add(
    store.streams.color_image_embedded
        .search(search_vector)
        .order_by("ts")
        .map(lambda obs: obs.derive(data=obs.similarity))
        .transform(smooth(20)),
    label="door-ness",
    axis="semantics",
)


plot.to_svg("assets/plot_speed.svg")
```





![output](assets/plot_speed.svg)
