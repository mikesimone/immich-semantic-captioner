# Known issues

## MP4 container declares the wrong duration -- videos are under-sampled

Some files declare a duration in their MP4 header that is far shorter than the actual
content. `ffprobe -show_entries format=duration` reports the header value, and Immich stores
it, so both agree on a number that is simply wrong.

Confirmed case:

    "Cream Filled #1 No Music.mp4"   (asset e8e67cc3-b39d-4df9-aea5-4f6d8a27dca0)
      container header : 409.34s  (6:49)
      actual decoded   : 1268.30s (21:08)   -- 38,012 frames @ 29.97fps

`probe_duration_seconds()` returns the header value, and both
`compute_video_timestamps()` and `compute_dense_timestamps()` build their sample points from
it. For the file above the captioner only ever looked at the first 6:49 of 21:08 -- it never
saw two thirds of the video. Any creampie, bondage, species or lactation signal in the
remaining 14 minutes was invisible, and dense mode's frame budget was spent on a fraction of
the runtime.

This is a plausible contributor to odd counts on long compilations specifically, since
they're the files most likely to have been re-muxed.

### Suggested fix

In `probe_duration_seconds()`, cross-check the header against the frame count:

    ffprobe -v error -select_streams v:0 \
      -show_entries stream=nb_frames,r_frame_rate \
      -show_entries format=duration -of default=noprint_wrappers=1 FILE

If `nb_frames / r_frame_rate` disagrees with `format=duration` by more than a few percent,
trust the frame count. No decode required, so it's cheap enough to run for every video.

Afterwards, survey the library for affected files and clear their descriptions so they get
re-captioned across their full runtime.

### Related

The annotator in the `porn-classifier` repo already works around the display half of this:
it shows the larger of the browser's decoded duration and Immich's metadata, and says so
when playback passes the declared end. See `TODO.md` there.
