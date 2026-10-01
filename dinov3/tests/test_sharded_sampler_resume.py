import itertools
import unittest
import warnings

from dinov3.data.samplers import ShardedInfiniteSampler


class ShardedSamplerResumeTests(unittest.TestCase):
    def test_resume_matches_consuming_each_rank_stream(self):
        for shuffle in (False, True):
            for new_shuffle in (False, True):
                for size, world_size in ((8, 2), (11, 3), (9, 4), (8, 1)):
                    for rank in range(world_size):
                        for advance in (0, 1, 3, 8, 13, 31):
                            with self.subTest(
                                shuffle=shuffle,
                                new=new_shuffle,
                                size=size,
                                world=world_size,
                                rank=rank,
                                advance=advance,
                            ):
                                kwargs = dict(
                                    sample_count=size,
                                    shuffle=shuffle,
                                    seed=19,
                                    start=rank,
                                    step=world_size,
                                    use_new_shuffle_tensor_slice=new_shuffle,
                                )
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    uninterrupted = list(
                                        itertools.islice(ShardedInfiniteSampler(**kwargs), advance, advance + 20)
                                    )
                                    resumed = list(
                                        itertools.islice(ShardedInfiniteSampler(**kwargs, advance=advance), 20)
                                    )
                                self.assertEqual(resumed, uninterrupted)

    def test_ordered_uneven_shards_keep_their_actual_cycle_lengths(self):
        for rank, expected in ((0, [0, 3, 6]), (1, [1, 4]), (2, [2, 5])):
            sampler = ShardedInfiniteSampler(sample_count=7, shuffle=False, start=rank, step=3, advance=7)
            actual = list(itertools.islice(sampler, 12))
            self.assertEqual(actual, [expected[i % len(expected)] for i in range(7, 19)])


if __name__ == "__main__":
    unittest.main()
