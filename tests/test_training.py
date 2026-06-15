import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from training.event_postprocess import (
    event_metrics,
    predictions_to_events,
    timeline_to_events,
)
from training.folds import (
    annotate_groups,
    boundary_event_ids,
    make_fold_assignment,
)
from training.normalization import StationRobustNormalizer
from training.pipeline import load_locked_postprocess
from training.sampler import BalancedGroupBatchSampler


def _window(
    sample_id: str,
    station: str,
    center_time: str,
    label: int,
    event_id: str = "",
    source_data: str = "continuous/AAA/2008_data.npy",
    start_offset: int = 0,
    length: int = 4,
) -> dict:
    center = pd.Timestamp(center_time)
    return {
        "sample_id": sample_id,
        "station": station,
        "window_start": center - pd.Timedelta(hours=3),
        "window_end": center + pd.Timedelta(hours=3),
        "center_time": center,
        "source_data": source_data,
        "start_offset": start_offset,
        "length": length,
        "label": label,
        "event_id": event_id,
        "symh_min": -60.0 if label == 1 else -5.0,
        "partition_year": center.year,
    }


class FoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events = pd.DataFrame(
            [
                {
                    "event_id": "storm_cross_year",
                    "event_start": pd.Timestamp("2011-12-31 22:00"),
                    "event_end": pd.Timestamp("2012-01-01 04:00"),
                    "symh_min": -80,
                    "duration_minutes": 360,
                },
                {
                    "event_id": "storm_boundary",
                    "event_start": pd.Timestamp("2019-12-31 23:00"),
                    "event_end": pd.Timestamp("2020-01-01 02:00"),
                    "symh_min": -70,
                    "duration_minutes": 180,
                },
            ]
        )

    def test_cross_year_event_uses_event_start_year(self) -> None:
        index = pd.DataFrame(
            [
                _window(
                    "storm_a",
                    "AAA",
                    "2011-12-31 23:00",
                    1,
                    "storm_cross_year",
                ),
                _window(
                    "storm_b",
                    "BBB",
                    "2012-01-01 01:00",
                    1,
                    "storm_cross_year",
                ),
                _window("quiet_a", "AAA", "2011-06-01 12:00", 0),
                _window("quiet_b", "BBB", "2012-06-01 12:00", 0),
            ]
        )
        annotated = annotate_groups(index, self.events)
        event_rows = annotated.loc[annotated["event_id"].eq("storm_cross_year")]
        self.assertEqual(set(event_rows["group_year"]), {2011})

        fold = make_fold_assignment(
            annotated,
            self.events,
            development_years=[2011, 2012],
            final_test_years=[2013],
            validation_years=[2011],
            name="fold_test",
        )
        self.assertEqual(
            set(fold.validation.loc[fold.validation["label"].eq(1), "event_id"]),
            {"storm_cross_year"},
        )
        self.assertFalse(fold.train["event_id"].eq("storm_cross_year").any())
        self.assertEqual(fold.audit["event_id_leakage_count"], 0)

    def test_final_boundary_event_is_excluded(self) -> None:
        excluded = boundary_event_ids(
            self.events,
            development_years=list(range(2008, 2020)),
            final_test_years=[2020, 2021, 2022],
        )
        self.assertEqual(excluded, {"storm_boundary"})

        index = pd.DataFrame(
            [
                _window(
                    "boundary_2019",
                    "AAA",
                    "2019-12-31 23:30",
                    1,
                    "storm_boundary",
                ),
                _window(
                    "boundary_2020",
                    "BBB",
                    "2020-01-01 00:30",
                    1,
                    "storm_boundary",
                ),
                _window("quiet_2019", "AAA", "2019-06-01 12:00", 0),
            ]
        )
        annotated = annotate_groups(index, self.events)
        fold = make_fold_assignment(
            annotated,
            self.events,
            development_years=list(range(2008, 2020)),
            final_test_years=[2020, 2021, 2022],
            validation_years=[2019],
            name="fold_3",
        )
        self.assertFalse(fold.validation["event_id"].eq("storm_boundary").any())
        self.assertEqual(
            fold.audit["excluded_boundary_event_ids"],
            ["storm_boundary"],
        )

    def test_nonstorm_day_is_a_single_group_across_stations(self) -> None:
        index = pd.DataFrame(
            [
                _window("a", "AAA", "2011-05-06 01:00", 0),
                _window("b", "BBB", "2011-05-06 19:00", 0),
                _window("c", "AAA", "2012-05-06 01:00", 0),
            ]
        )
        annotated = annotate_groups(index, self.events)
        same_day = annotated.loc[
            annotated["center_time"].dt.strftime("%Y%m%d").eq("20110506")
        ]
        self.assertEqual(set(same_day["group_id"]), {"nonstorm_20110506"})

        fold = make_fold_assignment(
            annotated,
            self.events,
            development_years=[2011, 2012],
            final_test_years=[2013],
            validation_years=[2011],
            name="fold_test",
        )
        self.assertEqual(fold.audit["time_group_id_leakage_count"], 0)
        self.assertEqual(len(fold.validation), 2)


class NormalizationTests(unittest.TestCase):
    def test_fit_uses_only_training_groups_and_transform_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "continuous" / "AAA" / "2008_data.npy"
            data_path.parent.mkdir(parents=True)
            raw = np.array(
                [
                    [0, 10, 100],
                    [1, 11, 101],
                    [2, 12, 102],
                    [3, 13, 103],
                    [10, 20, 110],
                    [11, 21, 111],
                    [12, 22, 112],
                    [13, 23, 113],
                ],
                dtype=np.float32,
            )
            np.save(data_path, raw)
            train_index = pd.DataFrame(
                [
                    _window(
                        "train_1",
                        "AAA",
                        "2008-01-01 03:00",
                        0,
                        source_data="continuous/AAA/2008_data.npy",
                        start_offset=0,
                        length=4,
                    ),
                    _window(
                        "train_2",
                        "AAA",
                        "2008-01-02 03:00",
                        1,
                        "storm_train",
                        source_data="continuous/AAA/2008_data.npy",
                        start_offset=4,
                        length=4,
                    ),
                ]
            )
            train_index["group_year"] = 2008
            normalizer = StationRobustNormalizer.fit(
                dataset_root=root,
                training_index=train_index,
                training_years=[2008],
                baseline_minutes=2,
                max_windows_per_station=10,
                global_max_points_per_station_channel=100,
                seed=42,
                epsilon=1.0e-6,
                clip=2.0,
            )
            self.assertEqual(normalizer.metadata["training_years"], [2008])
            self.assertEqual(normalizer.station_stats["AAA"]["sampled_windows"], 2)
            transformed = normalizer.transform(
                np.full((3, 4), 1.0e6, dtype=np.float32),
                "AAA",
            )
            self.assertTrue(np.isfinite(transformed).all())
            self.assertLessEqual(float(np.abs(transformed).max()), 2.0)

            invalid_index = train_index.copy()
            invalid_index.loc[1, "group_year"] = 2020
            with self.assertRaisesRegex(ValueError, "non-training years"):
                StationRobustNormalizer.fit(
                    dataset_root=root,
                    training_index=invalid_index,
                    training_years=[2008],
                    baseline_minutes=2,
                    max_windows_per_station=10,
                    global_max_points_per_station_channel=100,
                    seed=42,
                    epsilon=1.0e-6,
                    clip=2.0,
                )


class SamplerTests(unittest.TestCase):
    def test_batches_are_balanced_and_groups_are_sampled_first(self) -> None:
        rows = []
        for index in range(100):
            rows.append(
                _window(
                    f"long_{index}",
                    "AAA",
                    f"2011-01-{1 + index // 20:02d} {index % 20:02d}:00",
                    1,
                    "long_event",
                )
            )
        rows.append(
            _window(
                "short",
                "BBB",
                "2011-02-01 00:00",
                1,
                "short_event",
            )
        )
        for hour in range(20):
            rows.append(
                _window(
                    f"quiet_a_{hour}",
                    "AAA",
                    f"2011-03-01 {hour:02d}:00",
                    0,
                )
            )
        rows.append(_window("quiet_b", "BBB", "2011-03-02 00:00", 0))
        frame = pd.DataFrame(rows).reset_index(drop=True)
        sampler = BalancedGroupBatchSampler(
            frame,
            batch_size=20,
            samples_per_epoch=10000,
            seed=7,
        )
        event_counts = {"long_event": 0, "short_event": 0}
        date_counts = {"20110301": 0, "20110302": 0}
        for batch in sampler:
            labels = frame.iloc[batch]["label"].to_numpy()
            self.assertEqual(int((labels == 1).sum()), 10)
            self.assertEqual(int((labels == 0).sum()), 10)
            selected = frame.iloc[batch]
            for event_id in selected.loc[selected["label"].eq(1), "event_id"]:
                event_counts[event_id] += 1
            for center in selected.loc[selected["label"].eq(0), "center_time"]:
                date_counts[pd.Timestamp(center).strftime("%Y%m%d")] += 1
        event_ratio = event_counts["long_event"] / sum(event_counts.values())
        date_ratio = date_counts["20110301"] / sum(date_counts.values())
        self.assertTrue(0.45 <= event_ratio <= 0.55)
        self.assertTrue(0.45 <= date_ratio <= 0.55)


class EventPostprocessTests(unittest.TestCase):
    def test_short_gap_candidates_are_merged(self) -> None:
        timeline = pd.DataFrame(
            {
                "center_time": pd.to_datetime(
                    ["2021-01-01 00:00", "2021-01-01 03:00"]
                ),
                "score": [0.9, 0.8],
                "supporting_stations": [["AAA"], ["BBB"]],
            }
        )
        events = timeline_to_events(
            timeline,
            threshold=0.5,
            merge_gap_hours=3,
            stride_hours=1,
            boundary_half_width_hours=0.5,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]["supporting_stations"], ["AAA", "BBB"])

    def test_single_station_inference_degrades_minimum_station_count(self) -> None:
        predictions = pd.DataFrame(
            {
                "station": ["AAA", "AAA", "AAA"],
                "center_time": pd.date_range(
                    "2021-01-01 00:00",
                    periods=3,
                    freq="1h",
                ),
                "score": [0.8, 0.9, 0.8],
            }
        )
        events, fused, _ = predictions_to_events(
            predictions,
            threshold=0.5,
            smoothing_points=1,
            merge_gap_hours=3,
            min_stations=3,
        )
        self.assertEqual(len(fused), 3)
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]["supporting_stations"], ["AAA"])

    def test_event_metrics_report_detection_by_year(self) -> None:
        references = pd.DataFrame(
            {
                "event_id": ["a", "b"],
                "event_start": pd.to_datetime(
                    ["2020-01-01 00:00", "2021-01-01 00:00"]
                ),
                "event_end": pd.to_datetime(
                    ["2020-01-01 06:00", "2021-01-01 06:00"]
                ),
            }
        )
        predicted = pd.DataFrame(
            {
                "event_start": pd.to_datetime(["2020-01-01 01:00"]),
                "event_end": pd.to_datetime(["2020-01-01 05:00"]),
                "maximum_probability": [0.9],
                "supporting_stations": [["AAA"]],
            }
        )
        metrics, _ = event_metrics(predicted, references)
        self.assertEqual(
            metrics["detection_by_year"]["2020"]["detected_event_count"],
            1,
        )
        self.assertEqual(
            metrics["detection_by_year"]["2021"]["missed_event_count"],
            1,
        )


class TorchTrainingTests(unittest.TestCase):
    @unittest.skipUnless(
        __import__("importlib").util.find_spec("torch") is not None,
        "PyTorch is not installed in this local environment.",
    )
    def test_cnn_shape_and_fixed_epoch_checkpoint_resume(self) -> None:
        import torch

        from training.models import CNN1D
        from training.trainer import train_fixed_epochs

        class TinyDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return 8

            def __getitem__(self, item: int):
                return {
                    "input": torch.full((3, 360), float(item) / 10),
                    "label": torch.tensor(float(item % 2)),
                    "position": item,
                }

        model = CNN1D(input_channels=3, dropout=0.1)
        self.assertEqual(tuple(model(torch.zeros(2, 3, 360)).shape), (2,))
        loader = torch.utils.data.DataLoader(TinyDataset(), batch_size=4)
        config = {
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "amp": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            first = train_fixed_epochs(
                model,
                loader,
                temporary,
                config,
                {"input_channels": 3, "dropout": 0.1},
                epochs=1,
                device=torch.device("cpu"),
                resume=False,
            )
            self.assertEqual(len(first["history"]), 1)
            resumed = train_fixed_epochs(
                CNN1D(input_channels=3, dropout=0.1),
                loader,
                temporary,
                config,
                {"input_channels": 3, "dropout": 0.1},
                epochs=2,
                device=torch.device("cpu"),
                resume=True,
            )
            self.assertEqual(len(resumed["history"]), 2)
            self.assertTrue((Path(temporary) / "final_model.pt").exists())


class FinalTestLockTests(unittest.TestCase):
    def test_final_test_accepts_only_complete_oof_locked_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "selected_postprocess.json"
            payload = {
                "threshold": 0.5,
                "smoothing_points": 3,
                "merge_gap_hours": 3,
                "fusion": "median",
                "min_stations": 3,
                "stride_hours": 1,
                "boundary_half_width_hours": 0.5,
                "match_tolerance_hours": 0,
                "selection_source": "out_of_fold_only",
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_locked_postprocess(root)["threshold"], 0.5)

            payload["selection_source"] = "final_test_search"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not selected from OOF"):
                load_locked_postprocess(root)


if __name__ == "__main__":
    unittest.main()
