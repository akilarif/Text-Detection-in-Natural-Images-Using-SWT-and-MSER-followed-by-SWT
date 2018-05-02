# -*- encoding: utf-8 -*-
from __future__ import division
from collections import defaultdict
import hashlib
import math
import os
import time
from urllib2 import urlopen

import numpy as np
import cv2
import scipy.sparse, scipy.spatial

import progressbar

t0 = time.clock()

diagnostics = True

#local_filename = "image1.jpg"
local_filename = "IMG_2488.jpg"

darkTextOnLightBackground = 1

#Your image path i-e receipt path
img = cv2.imread(local_filename)
#img = cv2.imread('img2.JPG')

print("MSER Started")

#Create MSER object
mser = cv2.MSER_create()

#Convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("gray.jpg", gray)

img_copy = img.copy()

#detect regions in gray scale image
regions, bboxes = mser.detectRegions(gray)
#print bboxes

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv2.polylines(img_copy, hulls, 1, (0, 255, 0))
#cv2.polylines(img, hulls, 1, (0, 255, 0))

cv2.imwrite('mser.jpg', img_copy)
#cv2.imwrite('mser.jpg', img)

mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

for contour in hulls:

    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

#this is used to find only text regions, remaining are ignored
text_only = cv2.bitwise_and(img, img, mask=mask)

cv2.imwrite("result.jpg", text_only)

print("MSER Ended")
print time.clock() - t0

class SWTScrubber(object):
    @classmethod
    def scrub(cls, filepath, regions, bboxes, darkTextOnLightBackground=1):
        """
        Apply Stroke-Width Transform to image.

        :param filepath: relative or absolute filepath to source image
        :return: numpy array representing result of transform
        """

        if darkTextOnLightBackground == 0:  # Detecting Light Text on Dark Background
            searchDirection = -1
        else:
            searchDirection = 1

        canny, sobelx, sobely, theta = cls._create_derivative(filepath)

        swt = np.empty(theta.shape)
        swt[:] = np.Infinity
        bar = progressbar.ProgressBar(maxval=len(regions), widgets=[progressbar.Bar(marker='=', left='[', right=']'), ' ', progressbar.SimpleProgress()])
        bar.start()
        for i, region in enumerate(regions):
            bar.update(i + 1)
            xx, yy, w, h = bboxes[i]
            swt_region = cls._swt(theta, canny, sobelx, sobely, (xx, yy, w, h), searchDirection)
            swt[swt == np.Infinity] = 0.
            swt_region[swt_region == np.Infinity] = 0.
            swt = np.add(swt, swt_region)
            swt[swt == 0.] = np.Infinity
        
        #print swt
        shapes = cls._connect_components(swt)
        swts, heights, widths, topleft_pts, images = cls._find_letters(swt, shapes)
        word_images = cls._find_words(swts, heights, widths, topleft_pts, images)

        final_mask = np.zeros(swt.shape)
        for word in word_images:
            final_mask += word
        return final_mask

    @classmethod
    def _create_derivative(cls, filepath):
        img = cv2.imread(filepath,0)
        edges = cv2.Canny(img, 175, 320, apertureSize=3)

        '''sigma = 0.33
        v = np.median(img)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(img, lower, upper)'''

        # Create gradient map using Sobel
        sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=-1)
        sobely64f = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=-1)

        theta = np.arctan2(sobely64f, sobelx64f)
        if diagnostics:
            cv2.imwrite('edges.jpg',edges)
            cv2.imwrite('sobelx64f.jpg', np.absolute(sobelx64f))
            cv2.imwrite('sobely64f.jpg', np.absolute(sobely64f))
            # amplify theta for visual inspection
            theta_visible = (theta + np.pi)*255/(2*np.pi)
            cv2.imwrite('theta.jpg', theta_visible)
        return (edges, sobelx64f, sobely64f, theta)

    @classmethod
    def _swt(self, theta, edges, sobelx64f, sobely64f, (xx, yy, w, h), searchDirection):
        # create empty image, initialized to infinity
        swt = np.empty(theta.shape)
        swt[:] = np.Infinity
        rays = []

        #print time.clock() - t0

        # now iterate over pixels in image, checking Canny to see if we're on an edge.
        # if we are, follow a normal a ray to either the next edge or image border
        # edgesSparse = scipy.sparse.coo_matrix(edges)
        step_x_g = -1 * sobelx64f
        step_y_g = -1 * sobely64f
        mag_g = np.sqrt( step_x_g * step_x_g + step_y_g * step_y_g )
        #grad_x_g = step_x_g / mag_g
        grad_x_g = np.true_divide(step_x_g, mag_g, where= (step_x_g != 0) | (mag_g != 0))
        #grad_y_g = step_y_g / mag_g
        grad_y_g = np.true_divide(step_y_g, mag_g, where= (step_y_g != 0) | (mag_g != 0))

        #for x in xrange(edges.shape[1]):
        for x in xrange(xx, xx + w):
            #for y in xrange(edges.shape[0]):
            for y in xrange(yy, yy + h):
                if edges[y, x] > 0:
                    step_x = step_x_g[y, x]
                    step_y = step_y_g[y, x]
                    mag = mag_g[y, x]
                    grad_x = grad_x_g[y, x]
                    grad_y = grad_y_g[y, x]
                    ray = []
                    ray.append((x, y))
                    prev_x, prev_y, i = x, y, 0
                    while True:
                        i += 1

                        if grad_x == 0 and grad_y == 0:
                            print "#############"
                            break

                        #cur_x = math.floor(x + grad_x * i)
                        cur_x = np.int(np.floor(x + searchDirection * grad_x * i))
                        #cur_y = math.floor(y + grad_y * i)
                        cur_y = np.int(np.floor(y + searchDirection * grad_y * i))

                        if cur_x != prev_x or cur_y != prev_y:
                            # we have moved to the next pixel!
                            try:
                                if edges[cur_y, cur_x] > 0:
                                    # found edge,
                                    ray.append((cur_x, cur_y))
                                    theta_point = theta[y, x]
                                    alpha = theta[cur_y, cur_x]
                                    '''grad_dot_grad_g = grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x]
                                    mod_grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
                                    mod_grad_g = np.sqrt((-grad_x_g[cur_y, cur_x]) ** 2 + (-grad_y_g[cur_y, cur_x]) ** 2)
                                    cos_theta = np.true_divide(grad_dot_grad_g, mod_grad * mod_grad_g, where= (grad_dot_grad_g != 0) | (mod_grad != 0) | (mod_grad_g != 0))
                                    
                                    if float(cos_theta) <= -1.0:
                                    	print "BREAK"
                                    	break
                                    if math.acos(float(cos_theta)) < np.pi/6.0:'''

                                    # Math domain error was given on the line with acos() even if the argument is -1.0 or 1.0
                                    # So the if statement below was written to correct this
                                    if (grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x]) <= -1.0 or grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x] >= 1.0:
                                        break
                                    
                                    if np.arccos(grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x]) < np.pi/2.0:
                                    	thickness = np.sqrt((cur_x - x) ** 2 + (cur_y - y) **2)
                                        for (rp_x, rp_y) in ray:
                                            swt[rp_y, rp_x] = min(thickness, swt[rp_y, rp_x])
                                        rays.append(ray)
                                    break
                                # this is positioned at end to ensure we don't add a point beyond image boundary
                                ray.append((cur_x, cur_y))
                            except IndexError:
                                # reached image boundary
                                break
                            prev_x = cur_x
                            prev_y = cur_y

        # Compute median SWT

        for ray in rays:
            median = np.median([swt[y, x] for (x, y) in ray])
            for (x, y) in ray:
                swt[y, x] = min(median, swt[y, x])
        if diagnostics:
            cv2.imwrite('swt.jpg', swt * 100)
        
        return swt

    @classmethod
    def _connect_components(cls, swt):
        # STEP: Compute distinct connected components
        # Implementation of disjoint-set
        class Label(object):
            def __init__(self, value):
                self.value = value
                self.parent = self
                self.rank = 0
            def __eq__(self, other):
                if type(other) is type(self):
                    return self.value == other.value
                else:
                    return False
            def __ne__(self, other):
                return not self.__eq__(other)

        ld = {}

        def MakeSet(x):
            try:
                return ld[x]
            except KeyError:
                item = Label(x)
                ld[x] = item
                return item

        def Find(item):
            # item = ld[x]
            if item.parent != item:
                item.parent = Find(item.parent)
            return item.parent

        def Union(x, y):
            """
            :param x:
            :param y:
            :return: root node of new union tree
            """
            x_root = Find(x)
            y_root = Find(y)
            if x_root == y_root:
                return x_root

            if x_root.rank < y_root.rank:
                x_root.parent = y_root
                return y_root
            elif x_root.rank > y_root.rank:
                y_root.parent = x_root
                return x_root
            else:
                y_root.parent = x_root
                x_root.rank += 1
                return x_root

        # apply Connected Component algorithm, comparing SWT values.
        # components with a SWT ratio less extreme than 1:3 are assumed to be
        # connected. Apply twice, once for each ray direction/orientation, to
        # allow for dark-on-light and light-on-dark texts
        trees = {}
        # Assumption: we'll never have more than 65535-1 unique components
        label_map = np.zeros(shape=swt.shape, dtype=np.uint16)
        next_label = 1
        # First Pass, raster scan-style
        swt_ratio_threshold = 3.0
        for y in xrange(swt.shape[0]):
            for x in xrange(swt.shape[1]):
                sw_point = swt[y, x]
                if sw_point < np.Infinity and sw_point > 0:
                    neighbors = [(y, x-1),   # west
                                 (y-1, x-1), # northwest
                                 (y-1, x),   # north
                                 (y-1, x+1)] # northeast
                    connected_neighbors = None
                    neighborvals = []

                    for neighbor in neighbors:
                        # west
                        try:
                            sw_n = swt[neighbor]
                            label_n = label_map[neighbor]
                        except IndexError:
                            continue
                        if label_n > 0 and sw_n / sw_point < swt_ratio_threshold and sw_point / sw_n < swt_ratio_threshold:
                            neighborvals.append(label_n)
                            if connected_neighbors:
                                connected_neighbors = Union(connected_neighbors, MakeSet(label_n))
                            else:
                                connected_neighbors = MakeSet(label_n)

                    if not connected_neighbors:
                        # We don't see any connections to North/West
                        trees[next_label] = (MakeSet(next_label))
                        label_map[y, x] = next_label
                        next_label += 1
                    else:
                        # We have at least one connection to North/West
                        label_map[y, x] = min(neighborvals)
                        # For each neighbor, make note that their respective connected_neighbors are connected
                        # for label in connected_neighbors. @todo: do I need to loop at all neighbor trees?
                        trees[connected_neighbors.value] = Union(trees[connected_neighbors.value], connected_neighbors)

        # Second pass. re-base all labeling with representative label for each connected tree
        layers = {}
        contours = defaultdict(list)
        for x in xrange(swt.shape[1]):
            for y in xrange(swt.shape[0]):
                if label_map[y, x] > 0:
                    item = ld[label_map[y, x]]
                    common_label = Find(item).value
                    label_map[y, x] = common_label
                    contours[common_label].append([x, y])
                    try:
                        layer = layers[common_label]
                    except KeyError:
                        layers[common_label] = np.zeros(shape=swt.shape, dtype=np.uint16)
                        layer = layers[common_label]

                    layer[y, x] = 1

        return layers

    @classmethod
    def _find_letters(cls, swt, shapes):
        # STEP: Discard shapes that are probably not letters
        swts = []
        heights = []
        widths = []
        topleft_pts = []
        images = []

        for label,layer in shapes.iteritems():
            (nz_y, nz_x) = np.nonzero(layer)
            east, west, south, north = max(nz_x), min(nz_x), max(nz_y), min(nz_y)
            width, height = east - west, south - north

            '''if np.var(swt[(nz_y, nz_x)]) > np.mean(swt[(nz_y, nz_x)])/2.0:
            	continue'''

            if width < 8 or height < 8:
                continue

            if width / height > 10 or height / width > 10:
                continue

            diameter = math.sqrt(width * width + height * height)
            median_swt = np.median(swt[(nz_y, nz_x)])
            if diameter / median_swt > 10:
                continue

            if width / layer.shape[1] > 0.4 or height / layer.shape[0] > 0.4:
                continue

            '''if diagnostics:
                print " written to image."
                cv2.imwrite('layer'+ str(label) +'.jpg', layer * 255)'''

            # we use log_base_2 so we can do linear distance comparison later using k-d tree
            # ie, if log2(x) - log2(y) > 1, we know that x > 2*y
            # Assumption: we've eliminated anything with median_swt == 1
            swts.append([math.log(median_swt, 2)])
            heights.append([math.log(height, 2)])
            topleft_pts.append(np.asarray([north, west]))
            widths.append(width)
            images.append(layer)

        return swts, heights, widths, topleft_pts, images

    @classmethod
    def _find_words(cls, swts, heights, widths, topleft_pts, images):
        # Find all shape pairs that have similar median stroke widths
        print 'SWTS'
        print swts
        print 'DONESWTS'
        swt_tree = scipy.spatial.KDTree(np.asarray(swts))
        stp = swt_tree.query_pairs(1)

        # Find all shape pairs that have similar heights
        height_tree = scipy.spatial.KDTree(np.asarray(heights))
        htp = height_tree.query_pairs(1)

        # Intersection of valid pairings
        isect = htp.intersection(stp)

        chains = []
        pairs = []
        pair_angles = []
        for pair in isect:
            left = pair[0]
            right = pair[1]
            widest = max(widths[left], widths[right])
            distance = np.linalg.norm(topleft_pts[left] - topleft_pts[right])
            if distance < widest * 3:
                delta_yx = topleft_pts[left] - topleft_pts[right]
                angle = np.arctan2(delta_yx[0], delta_yx[1])
                if angle < 0:
                    angle += np.pi

                pairs.append(pair)
                pair_angles.append(np.asarray([angle]))

        angle_tree = scipy.spatial.KDTree(np.asarray(pair_angles))
        atp = angle_tree.query_pairs(np.pi/12)

        for pair_idx in atp:
            pair_a = pairs[pair_idx[0]]
            pair_b = pairs[pair_idx[1]]
            left_a = pair_a[0]
            right_a = pair_a[1]
            left_b = pair_b[0]
            right_b = pair_b[1]

            # @todo - this is O(n^2) or similar, extremely naive. Use a search tree.
            added = False
            for chain in chains:
                if left_a in chain:
                    chain.add(right_a)
                    added = True
                elif right_a in chain:
                    chain.add(left_a)
                    added = True
            if not added:
                chains.append(set([left_a, right_a]))
            added = False
            for chain in chains:
                if left_b in chain:
                    chain.add(right_b)
                    added = True
                elif right_b in chain:
                    chain.add(left_b)
                    added = True
            if not added:
                chains.append(set([left_b, right_b]))

        word_images = []
        #for chain in [c for c in chains if len(c) > 3]:
        for chain in [c for c in chains if len(c) >= 3]:
            for idx in chain:
                word_images.append(images[idx])
                # cv2.imwrite('keeper'+ str(idx) +'.jpg', images[idx] * 255)
                # final += images[idx]

        return word_images


#file_url = 'http://upload.wikimedia.org/wikipedia/commons/0/0b/ReceiptSwiss.jpg'
#local_filename = hashlib.sha224(file_url).hexdigest()

try:
    '''s3_response = urlopen(file_url)
    with open(local_filename, 'wb+') as destination:
        while True:
            # read file in 4kB chunks
            chunk = s3_response.read(4096)
            if not chunk: break
            destination.write(chunk)'''
    #final_mask = SWTScrubber.scrub('wallstreetsmd.jpeg')
    print("SWT Started")
    final_mask = SWTScrubber.scrub(local_filename, regions, bboxes, darkTextOnLightBackground)
    # final_mask = cv2.GaussianBlur(final_mask, (1, 3), 0)
    # cv2.GaussianBlur(sobelx64f, (3, 3), 0)
    cv2.imwrite('final.jpg', final_mask * 255)
    print("SWT Ended")
    print time.clock() - t0
finally:
    #s3_response.close()
    pass