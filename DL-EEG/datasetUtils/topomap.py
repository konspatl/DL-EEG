import numpy as np
import copy
import itertools
from mne import create_info
from mne.io.pick import (channel_type, pick_info,_pick_data_channels, )
from mne.utils import logger
from mne.channels.layout import _find_topomap_coords
from mne.io.meas_info import Info
from numpy.linalg import norm

from datasetUtils import EEG



def create_topomap(epoched_data, sfreq, channel_names, montage, pixel_res=10):
    """
    Creates a Topomap Representation of the EEG data

    Args:
        epoched_data, (ndarray): the epoched EEG data (epoch, channel, sample)
        sfreq (int): the sampling frequency
        channel_names (list): list with the EEG channel names
        montage (MNE structure): the montage of the EEG
        pixel_res (int): the pixel resolution of the topomap images (1 dimension)

    Returns:
        topomap_epoched_data (ndarray): the topomap epoched EEG data
    """

    print('Performing Topomap transformation...')

    topomap_epoched_data = np.empty((epoched_data.shape[0], pixel_res, pixel_res, epoched_data.shape[2]))

    orig_channel_names = channel_names.copy()
    # Check for channel_name inconsistencies (Change 10-20 names back to original)
    for i, channel in enumerate(orig_channel_names):
        if channel not in montage.ch_names:
            # GSN-HydroCel-256 , GSN-HydroCel-257
            if montage.kind in ['GSN-HydroCel-256', 'GSN-HydroCel-257']:
                orig_channel_names[i] = EEG.EGI257_10_20_inv_map[channel]
            # 'GSN-HydroCel-128' , 'GSN-HydroCel-129'
            elif montage.kind in ['GSN-HydroCel-128', 'GSN-HydroCel-129']:
                orig_channel_names[i] = EEG.EGI129_10_20_inv_map[channel]

    # Create Info Object
    info = create_info(orig_channel_names, sfreq, 'eeg', montage=montage)

    # MNE - Plot_Topomap Arguments
    data = np.asarray(epoched_data)
    pos = info
    outlines = 'head'
    extrapolate = 'head'
    head_pos = _find_10_20_head_pos(montage.kind)
    print('Head Pos: ' + str(head_pos))

    res = pixel_res

    data = np.asarray(data)
    logger.debug('Plotting topomap for data shape %s' % (data.shape,))

    if isinstance(pos, Info):  # infer pos from Info object
        picks = _pick_data_channels(pos)  # pick only data channels
        pos = pick_info(pos, picks)

        # check if there is only 1 channel type, and n_chans matches the data
        ch_type = {channel_type(pos, idx)
                   for idx, _ in enumerate(pos["chs"])}
        info_help = ("Pick Info with e.g. mne.pick_info and "
                     "mne.io.pick.channel_indices_by_type.")
        if len(ch_type) > 1:
            raise ValueError("Multiple channel types in Info structure. " +
                             info_help)
        elif len(pos["chs"]) != data.shape[1]:
            raise ValueError("Number of channels in the Info object and "
                             "the data array does not match. " + info_help)
        else:
            ch_type = ch_type.pop()

        if any(type_ in ch_type for type_ in ('planar', 'grad')):
            # deal with grad pairs
            from mne.channels.layout import (_merge_grad_data, find_layout,
                                             _pair_grad_sensors)

            picks, pos = _pair_grad_sensors(pos, find_layout(pos))
            data = _merge_grad_data(data[picks]).reshape(-1)
        else:
            picks = list(range(data.shape[1]))
            pos = _find_topomap_coords(pos, picks=picks)
            # (Cartesian (3D) to spherical, and from polar to cartesian (2D)


    # Give a helpful error message for common mistakes regarding the position
    # matrix.
    pos_help = ("Electrode positions should be specified as a 2D array with "
                "shape (n_channels, 2). Each row in this matrix contains the "
                "(x, y) position of an electrode.")
    if pos.ndim != 2:
        error = ("{ndim}D array supplied as electrode positions, where a 2D "
                 "array was expected").format(ndim=pos.ndim)
        raise ValueError(error + " " + pos_help)
    elif pos.shape[1] == 3:
        error = ("The supplied electrode positions matrix contains 3 columns. "
                 "Are you trying to specify XYZ coordinates? Perhaps the "
                 "mne.channels.create_eeg_layout function is useful for you.")
        raise ValueError(error + " " + pos_help)
    # No error is raised in case of pos.shape[1] == 4. In this case, it is
    # assumed the position matrix contains both (x, y) and (width, height)
    # values, such as Layout.pos.
    elif pos.shape[1] == 1 or pos.shape[1] > 4:
        raise ValueError(pos_help)

    if data.shape[1] != len(pos):
        raise ValueError("Data and pos need to be of same length. Got data of "
                         "length %s, pos of length %s" % (len(data), len(pos)))


    pos, outlines = _check_outlines(pos, outlines, head_pos)
    assert isinstance(outlines, dict)

    # find mask limits
    xlim = np.inf, -np.inf,
    ylim = np.inf, -np.inf,
    mask_ = np.c_[outlines['mask_pos']]
    xmin, xmax = (np.min(np.r_[xlim[0], mask_[:, 0]]),
                  np.max(np.r_[xlim[1], mask_[:, 0]]))
    ymin, ymax = (np.min(np.r_[ylim[0], mask_[:, 1]]),
                  np.max(np.r_[ylim[1], mask_[:, 1]]))

    # To interpolate the data, we multiply clip radius by 1.06 so that pixelated
    # edges of the interpolated image would appear under the mask
    head_radius = (None if extrapolate == 'local' else
                   outlines['clip_radius'][0] * 1.06)
    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    Xi, Yi = np.meshgrid(xi, yi)
    grid_data = _GridData(pos, extrapolate, head_radius)

    # Data Interpolation
    for epoch in range(epoched_data.shape[0]):
        for sample in range(epoched_data.shape[2]):
            data = epoched_data[epoch, :, sample]
            interp = grid_data.set_values(data)
            Zi = interp.set_locations(Xi, Yi)()
            topomap_epoched_data[epoch, :, :, sample] = Zi

    # Replace NaN values with 0
    np.nan_to_num(topomap_epoched_data, copy=False)

    return topomap_epoched_data


# ---------------------------------------------------UTIL FUNCTIONS-----------------------------------------------------

# Returns the head_pos (dict: center(x,y), scale(x,y)) for the topomaps, based on the 2D locations of the
# 10-20 channels of the given montage

def _find_10_20_head_pos(montage_name):
    channel_names = None
    if (montage_name in ['GSN-HydroCel-256', 'GSN-HydroCel-257']):
        channel_names = EEG.EGI257_10_20
    elif (montage_name in ['GSN-HydroCel-128', 'GSN-HydroCel-129']):
        channel_names = EEG.EGI129_10_20
    else:
        channel_names = EEG.EEG_10_20

    info_10_20 = create_info(channel_names, 1, 'eeg', montage=montage_name)
    pos = info_10_20

    # GET 2D Coordinates from pos Info object
    if isinstance(pos, Info):  # infer pos from Info object
        picks = _pick_data_channels(pos)  # pick only data channels
        pos = pick_info(pos, picks)
        pos = _find_topomap_coords(pos, picks=picks)
        # (Cartesian (3D) to spherical, and from polar to cartesian (2D)

    # COMPUTE Center and Scale, for head_pos of Topomap
    pos = np.array(pos, float)[:, :2]  # ensure we have a copy
    head_pos = dict()
    head_pos['center'] = 0.5 * (pos.max(axis=0) + pos.min(axis=0))  # Center on Average values
    head_pos['scale'] = 0.75 / (pos.max(axis=0) - pos.min(axis=0))  # Scale on 0.75/(max - min)

    return head_pos


# ---------------------------------------------------MNE FUNCTIONS------------------------------------------------------

def _check_outlines(pos, outlines, head_pos=None):
    """Check or create outlines for topoplot."""
    pos = np.array(pos, float)[:, :2]  # ensure we have a copy
    head_pos = dict() if head_pos is None else head_pos
    if not isinstance(head_pos, dict):
        raise TypeError('head_pos must be dict or None')
    head_pos = copy.deepcopy(head_pos)
    for key in head_pos.keys():
        if key not in ('center', 'scale'):
            raise KeyError('head_pos must only contain "center" and '
                           '"scale"')
        head_pos[key] = np.array(head_pos[key], float)
        if head_pos[key].shape != (2,):
            raise ValueError('head_pos["%s"] must have shape (2,), not '
                             '%s' % (key, head_pos[key].shape))

    if isinstance(outlines, np.ndarray) or outlines in ('head', 'skirt', None):
        radius = 0.5
        ll = np.linspace(0, 2 * np.pi, 101)
        head_x = np.cos(ll) * radius
        head_y = np.sin(ll) * radius
        nose_x = np.array([0.18, 0, -0.18]) * radius
        nose_y = np.array([radius - .004, radius * 1.15, radius - .004])
        ear_x = np.array([.497, .510, .518, .5299, .5419, .54, .547,
                          .532, .510, .489])
        ear_y = np.array([.0555, .0775, .0783, .0746, .0555, -.0055, -.0932,
                          -.1313, -.1384, -.1199])

        # shift and scale the electrode positions
        if 'center' not in head_pos:
            head_pos['center'] = 0.5 * (pos.max(axis=0) + pos.min(axis=0))
        pos -= head_pos['center']

        if outlines is not None:
            # Define the outline of the head, ears and nose
            outlines_dict = dict(head=(head_x, head_y), nose=(nose_x, nose_y),
                                 ear_left=(ear_x, ear_y),
                                 ear_right=(-ear_x, ear_y))
        else:
            outlines_dict = dict()

        if isinstance(outlines, str) and outlines == 'skirt':
            if 'scale' not in head_pos:
                # By default, fit electrodes inside the head circle
                head_pos['scale'] = 1.0 / (pos.max(axis=0) - pos.min(axis=0))
            pos *= head_pos['scale']

            # Make the figure encompass slightly more than all points
            mask_scale = 1.25 * (pos.max(axis=0) - pos.min(axis=0))

            outlines_dict['autoshrink'] = False
            outlines_dict['mask_pos'] = (mask_scale[0] * head_x,
                                         mask_scale[1] * head_y)
            outlines_dict['clip_radius'] = (mask_scale / 2.)
        else:
            if 'scale' not in head_pos:
                # The default is to make the points occupy a slightly smaller
                # proportion (0.85) of the total width and height
                # this number was empirically determined (seems to work well)
                head_pos['scale'] = 0.85 / (pos.max(axis=0) - pos.min(axis=0))
            pos *= head_pos['scale']
            outlines_dict['mask_pos'] = head_x, head_y
            if isinstance(outlines, np.ndarray):
                outlines_dict['autoshrink'] = False
                outlines_dict['clip_radius'] = outlines
                x_scale = np.max(outlines_dict['head'][0]) / outlines[0]
                y_scale = np.max(outlines_dict['head'][1]) / outlines[1]
                for key in ['head', 'nose', 'ear_left', 'ear_right']:
                    value = outlines_dict[key]
                    value = (value[0] / x_scale, value[1] / y_scale)
                    outlines_dict[key] = value
            else:
                outlines_dict['autoshrink'] = True
                outlines_dict['clip_radius'] = (0.5, 0.5)

        outlines = outlines_dict

    elif isinstance(outlines, dict):
        if 'mask_pos' not in outlines:
            raise ValueError('You must specify the coordinates of the image '
                             'mask.')
    else:
        raise ValueError('Invalid value for `outlines`.')

    return pos, outlines


class _GridData(object):
    """Unstructured (x,y) data interpolator.

    This class allows optimized interpolation by computing parameters
    for a fixed set of true points, and allowing the values at those points
    to be set independently.
    """

    def __init__(self, pos, method='box', head_radius=None):
        # in principle this works in N dimensions, not just 2
        assert pos.ndim == 2 and pos.shape[1] == 2
        # Adding points outside the extremes helps the interpolators
        outer_pts, tri = _get_extra_points(pos, method, head_radius)
        self.n_extra = outer_pts.shape[0]
        self.tri = tri

    def set_values(self, v):
        """Set the values at interpolation points."""
        # Rbf with thin-plate is what we used to use, but it's slower and
        # looks about the same:
        #
        #     zi = Rbf(x, y, v, function='multiquadric', smooth=0)(xi, yi)
        #
        # Eventually we could also do set_values with this class if we want,
        # see scipy/interpolate/rbf.py, especially the self.nodes one-liner.
        from scipy.interpolate import CloughTocher2DInterpolator
        v = np.concatenate((v, np.zeros(self.n_extra)))
        self.interpolator = CloughTocher2DInterpolator(self.tri, v)
        return self

    def set_locations(self, Xi, Yi):
        """Set locations for easier (delayed) calling."""
        self.Xi = Xi
        self.Yi = Yi
        return self

    def __call__(self, *args):
        """Evaluate the interpolator."""
        if len(args) == 0:
            args = [self.Xi, self.Yi]
        return self.interpolator(*args)


def _get_extra_points(pos, method, head_radius):
    """Get coordinates of additinal interpolation points.

    If head_radius is None, returns coordinates of convex hull of channel
    positions, expanded by the median inter-channel distance.
    Otherwise gives positions of points on the head circle placed with a step
    of median inter-channel distance.
    """
    from scipy.spatial.qhull import Delaunay

    # the old method of placement - large box
    if method == 'box':
        extremes = np.array([pos.min(axis=0), pos.max(axis=0)])
        diffs = extremes[1] - extremes[0]
        extremes[0] -= diffs
        extremes[1] += diffs
        eidx = np.array(list(itertools.product(
            *([[0] * (pos.shape[1] - 1) + [1]] * pos.shape[1]))))
        pidx = np.tile(np.arange(pos.shape[1])[np.newaxis], (len(eidx), 1))
        outer_pts = extremes[eidx, pidx]
        return outer_pts, Delaunay(np.concatenate((pos, outer_pts)))

    # check if positions are colinear:
    diffs = np.diff(pos, axis=0)
    with np.errstate(divide='ignore'):
        slopes = diffs[:, 1] / diffs[:, 0]
    colinear = ((slopes == slopes[0]).all() or np.isinf(slopes).all() or
                pos.shape[0] < 4)

    # compute median inter-electrode distance
    if colinear:
        dim = 1 if diffs[:, 1].sum() > diffs[:, 0].sum() else 0
        sorting = np.argsort(pos[:, dim])
        pos_sorted = pos[sorting, :]
        diffs = np.diff(pos_sorted, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        distance = np.median(distances)
    else:
        tri = Delaunay(pos, incremental=True)
        idx1, idx2, idx3 = tri.simplices.T
        distances = np.concatenate(
            [np.linalg.norm(pos[i1, :] - pos[i2, :], axis=1)
             for i1, i2 in zip([idx1, idx2], [idx2, idx3])])
        distance = np.median(distances)

    if method == 'local':
        if colinear:
            # special case for colinear points
            edge_points = sorting[[0, -1]]
            line_len = np.diff(pos[edge_points, :], axis=0)
            unit_vec = line_len / np.linalg.norm(line_len) * distance
            unit_vec_par = unit_vec[:, ::-1] * [[-1, 1]]

            edge_pos = (pos[edge_points, :] +
                        np.concatenate([-unit_vec, unit_vec], axis=0))
            new_pos = np.concatenate([pos + unit_vec_par,
                                      pos - unit_vec_par, edge_pos], axis=0)
            tri = Delaunay(np.concatenate([pos, new_pos], axis=0))
            return new_pos, tri

        # get the convex hull of data points from triangulation
        hull_pos = pos[tri.convex_hull]

        # extend the convex hull limits outwards a bit
        channels_center = pos.mean(axis=0, keepdims=True)
        radial_dir = hull_pos - channels_center[np.newaxis, :]
        unit_radial_dir = radial_dir / np.linalg.norm(radial_dir, axis=-1,
                                                      keepdims=True)
        hull_extended = hull_pos + unit_radial_dir * distance
        hull_diff = np.diff(hull_pos, axis=1)[:, 0]
        hull_distances = np.linalg.norm(hull_diff, axis=-1)

        # add points along hull edges so that the distance between points
        # is around that of average distance between channels
        add_points = list()
        eps = np.finfo('float').eps
        n_times_dist = np.round(hull_distances / distance).astype('int')
        for n in range(2, n_times_dist.max() + 1):
            mask = n_times_dist == n
            mult = np.arange(1 / n, 1 - eps, 1 / n)[:, np.newaxis, np.newaxis]
            steps = hull_diff[mask][np.newaxis, ...] * mult
            add_points.append((hull_extended[mask, 0][np.newaxis, ...] +
                               steps).reshape((-1, 2)))

        # remove duplicates from hull_extended
        #hull_extended = _remove_duplicate_rows(hull_extended.reshape((-1, 2)))
        new_pos = np.concatenate([hull_extended] + add_points)
    else:
        # return points on the head circle
        head_radius = 0.53 if head_radius is None else head_radius
        angle = np.arcsin(distance / 2 / head_radius) * 2
        points_l = np.arange(0, 2 * np.pi, angle)
        points_x = np.cos(points_l) * head_radius
        points_y = np.sin(points_l) * head_radius
        new_pos = np.stack([points_x, points_y], axis=1)
        if colinear:
            tri = Delaunay(np.concatenate([pos, new_pos], axis=0))
            return new_pos, tri
    tri.add_points(new_pos)
    return new_pos, tri


