"""Simple CLI entry points that call into the library Trainer/Predictor."""
import argparse
import pandas as pd
from pathlib import Path
from . import NeoRankTrainer, NeoRankPredictor, Config


def train_main(argv=None):
    parser = argparse.ArgumentParser(prog="neorank-train")
    parser.add_argument('--epitope', help='Path to epitope TSV', default=None)
    parser.add_argument('--tcell', help='Path to tcell TSV', default=None)
    parser.add_argument('--models-dir', help='Directory to save models (optional)', default=None)
    args = parser.parse_args(argv)

    trainer = NeoRankTrainer(epitope_file=args.epitope, tcell_file=args.tcell, config=Config())
    model_data = trainer.run()
    return 0


def predict_main(argv=None):
    parser = argparse.ArgumentParser(prog="neorank-predict")
    parser.add_argument('input_csv', help='Input CSV file containing peptide,HLA')
    parser.add_argument('--model', help='Path to model pickle (optional)', default=None)
    parser.add_argument('--output', help='Output CSV path (optional)', default=None)
    args = parser.parse_args(argv)

    predictor = NeoRankPredictor(model_path=args.model, config=Config())
    out_path = predictor.predict_csv(args.input_csv, args.output)
    print(f"Predictions written to: {out_path}")
    return 0
