# Text Detection in Natural Images

Two approaches are implemented in this repository to perform text localization- SWT, and MSER detection followed by SWT.

Steps in SWT algorithm:
=======================

1. Use OpenCV to extract image edges using Canny edge detection.

2. Calculate the x- and y-derivatives of the image, which can be superimposed
to calculate the image gradient. The gradient describes, for each pixel, the
direction of greatest contrast. In the case of an edge pixel, this is
synonymous with the vector normal to the edge.

3. For each edge pixel, traverse in the direction θ of the gradient until the
next edge pixel is encountered (or you fall off the image). If the
corresponding edge pixel's gradient is pointed in the opposite direction
(θ - π), we know the newly-encountered edge is roughly parallel to the
first, and we have just cut a slice (line) through a stroke. Record the stroke width,
in pixels, and assign this value to all pixels on the slice we just traversed.

4. For pixels that may belong to multiple lines, reconcile differences in those
line widths by assigning all pixels the median stroke width value. This allows 
the two strokes you might encounter in an 'L' shape to be considered with the
same, most common, stroke width.

5. Connect lines that overlap using a union-find (disjoint-set) data
structure, resulting in a disjoint set of all overlapping stroke slices. Each
set of lines is likely a single letter/character.

6. Apply some intelligent filtering to the line sets; we should eliminate
anything to small (width, height) to be a legible character, as well as
anything too long or fat (width:height ratio), or too sparse (diameter:stroke
width ratio) to realistically be a character.

7. Use a k-d tree to find pairings of similarly-stroked shapes (based on
stroke width), and intersect this with pairings of similarly-sized shapes
(based on height/width). Calculate the angle of the text between these two
characters (sloping upwards, downwards?).

8. Use a k-d tree to find pairings of letter pairings with similar
orientations. These groupings of letters likely form a word. Chain similar
pairs together.

9. Produce a final image containing the resulting words.

MSER Detection Followed by SWT
==============================
SWT is calculated only for the extremal regions detected by MSER Detection algorithm.
