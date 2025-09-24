"""stager

Implements the Stager class.

Additionally, when run as a script, a Stager instance is run once.
"""

import sys
import os
import traceback
import shutil

import pandas as pd
import json

# Allow running both as a module (python -m movie_rag.stager) and as a script
try:
    from . import utils
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from movie_rag import utils


class Stager:
    """Process the movie files in the landing area and store the completed files in the staging area.
    The files processed from the landing area are stored in an archive there.
    """
    def __init__(self, landing_area=None, staging_area=None):
        self.landing_area = landing_area
        self.staging_area = staging_area

        self.landing_archive = os.path.join(self.landing_area, 'archive')
        if not os.path.exists(self.landing_archive):
            os.makedirs(self.landing_archive)

        return

    def stage_file(self, fname):
        # Check if the output file already exists.
        staging_fname = os.sep.join([self.staging_area, fname])
        if os.path.exists(staging_fname):
            raise RuntimeError(f'Staged file already exists: {fname}.')

        # Read the movie file from the landing area.
        landing_fname = os.sep.join([self.landing_area, fname])
        with open(landing_fname, encoding=utils.determine_file_encoding(landing_fname, 1000)) as fnum:
            source_json = json.loads(fnum.read())

        # Two supported formats:
        # 1) Legacy format object with keys 'column_order' and 'tickets' (backwards compatibility)
        # 2) List of movie dictionaries (IMDB-like export)
        if isinstance(source_json, dict) and 'tickets' in source_json:
            cols = [c for c in source_json['column_order'].split('|') if c in source_json['tickets'][0]]
            df = pd.DataFrame(source_json['tickets'])[cols]
            # Clean up the tickets if present
            for col in ['user_mail_address', 'ciid_root_cause']:
                if col in df.columns:
                    del df[col]
        elif isinstance(source_json, list):
            # Normalize movie records into our canonical schema used downstream
            def _s(value):
                if value is None:
                    return ''
                try:
                    return str(value).strip()
                except Exception:
                    return ''

            normalized = []
            for rec in source_json:
                if not isinstance(rec, dict):
                    # Skip invalid entries
                    continue

                name = _s(rec.get('Title') or rec.get('name'))
                year_val = rec.get('Year') if 'Year' in rec else rec.get('year')
                try:
                    year_int = int(year_val) if year_val not in [None, ''] else 0
                except Exception:
                    year_int = 0

                director = _s(rec.get('Director') or rec.get('director'))
                description = _s(rec.get('Description') or rec.get('description'))

                rating_val = rec.get('Rating') if 'Rating' in rec else rec.get('rating')
                try:
                    rating_float = float(rating_val) if rating_val not in [None, ''] else 0.0
                except Exception:
                    rating_float = 0.0

                # Actors/Genre may be comma-separated strings or lists
                raw_cast = rec.get('Actors') if 'Actors' in rec else rec.get('cast', [])
                raw_genres = rec.get('Genre') if 'Genre' in rec else rec.get('genres', [])
                if isinstance(raw_cast, str):
                    cast_list = [s.strip() for s in raw_cast.split(',') if s.strip()]
                elif isinstance(raw_cast, list):
                    cast_list = [str(s).strip() for s in raw_cast if str(s).strip()]
                else:
                    cast_list = []
                if isinstance(raw_genres, str):
                    genres_list = [s.strip() for s in raw_genres.split(',') if s.strip()]
                elif isinstance(raw_genres, list):
                    genres_list = [str(s).strip() for s in raw_genres if str(s).strip()]
                else:
                    genres_list = []

                movie_id = rec.get('movie_id') or f"{name.lower().replace(' ', '_')}_{year_int}"
                normalized.append({
                    'movie_id': movie_id,
                    'name': name,
                    'year': year_int,
                    'description': description,
                    'director': director,
                    'cast': cast_list,
                    'genres': genres_list,
                    'rating': rating_float,
                })
            if not normalized:
                raise RuntimeError('No valid movie records found in landing JSON list.')
            df = pd.DataFrame(normalized)
        else:
            raise RuntimeError('Unsupported input JSON format in landing file.')

        # Write the staged JSON as a list of records
        df.to_json(staging_fname, orient='records', force_ascii=False)

        # Move the original movie file to the landing area's archive.
        landing_archive_fname = os.path.join(self.landing_archive, fname)
        shutil.move(landing_fname, landing_archive_fname)

        return

    def run(self):
        for fname in os.listdir(self.landing_area):
            if fname.endswith('.json'):
                print(f'Staging: {fname}')
                try:
                    self.stage_file(fname)
                except Exception:   # catch all Exceptions
                    # Print the error and continue with the rest of the files.
                    traceback.print_exc()
                    print(f'Error staging file: {fname}')

        return


if __name__ == '__main__':
    """Instantiate a Stager and run it once.
    """
    config = utils.get_config(sys.argv[1:])
    stager_config = {
        'landing_area': config.get('landing_area'),
        'staging_area': config.get('staging_area')
    }

    stager = Stager(**stager_config)
    stager.run()
