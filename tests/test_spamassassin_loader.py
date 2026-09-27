"""
Tests for the SpamAssassin dataset loader.
"""

from src.data.spamassassin_loader import load_spamassassin


def test_load_spamassassin():

    dataframe = load_spamassassin()


    assert len(dataframe) == 3302


    expected_columns = {
        "email_id",
        "source",
        "source_category",
        "sender",
        "receiver",
        "subject",
        "date",
        "body",
        "html_body",
        "attachments",
        "label",
    }

    assert expected_columns.issubset(
        dataframe.columns
    )


    assert set(dataframe["label"].unique()) == {0, 1}


    assert set(dataframe["source"].unique()) == {
        "spamassassin"
    }


    assert set(dataframe["source_category"].unique()) == {
        "easy_ham",
        "hard_ham",
        "spam",
    }


    assert (
        dataframe["label"].value_counts()[0]
        == 2801
    )

    assert (
        dataframe["label"].value_counts()[1]
        == 501
    )
