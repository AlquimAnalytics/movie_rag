"""prepper

Implements the classes:
  - MovieTablerCSV: prepare a DataFrame and store it in a CSV file
  - MovieTablerSQL: prepare a sqlite database
  - MovieEmbedder: prepare vector embeddings and store it in a sklearn vector store
  - MovieGrapher: prepare a knowledge graph in a Neo4j database
  - Prepper: coordinate the three preparer classes

Additionally, when run as a script, a Prepper instance is run once.
"""

import sys
import os
import traceback
import shutil

import pandas as pd
import json
import sqlite3
from neo4j import GraphDatabase

from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from . import utils

import time
import contextlib

@contextlib.contextmanager
def timer():
    """Simple timer context manager to replace PyLib timer."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        print(f"Execution time: {elapsed_time:.2f} seconds")


class MovieTablerCSV:
    def __init__(self, staging_area='', prepared_area='', table_store_fname=''):
        self.staging_area = staging_area
        self.prepared_area = prepared_area
        self.table_store_fname = table_store_fname

        return

    def prepare(self, fname):
        """Convert JSON data to a DataFrame and append it to a consolidated CSV file.
        """
        staging_fname = os.path.join(self.staging_area, fname)
        df = pd.DataFrame(json.load(open(staging_fname, "r")))
        df = df.replace('\n', ' ', regex=True)
        df = df.replace('\r', ' ', regex=True)

        prepared_fname = os.path.join(self.prepared_area, self.table_store_fname)
        df.to_csv(prepared_fname, sep='\t', index=False, mode='a')  # seperator \t is result of search

        return


class MovieTablerSQL:
    def __init__(self, staging_area='', prepared_area='', table_store_fname=''):
        self.staging_area = staging_area
        self.prepared_area = prepared_area
        self.table_store_fname = table_store_fname

        # Set up place-holders to avoid IDE warnings in open() and close().
        self.conn = None
        self.cursor = None

        return

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()
        return

    def open(self):
        """Open the sqlite cursor and connection.
        """
        # Ensure the prepared area exists
        os.makedirs(self.prepared_area, exist_ok=True)
        prepared_fname = os.path.join(self.prepared_area, self.table_store_fname)
        self.conn = sqlite3.connect(prepared_fname)
        self.cursor = self.conn.cursor()

        # Allow either existing tickets table or a movies table depending on incoming data
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS movies (
            movie_id TEXT PRIMARY KEY,
            name TEXT,
            year INTEGER,
            description TEXT,
            director TEXT,
            cast TEXT,
            genres TEXT,
            rating REAL
        )
        ''')

        # New normalized link tables for better analytics/filmography
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS movie_cast (
            movie_id TEXT,
            actor TEXT,
            PRIMARY KEY (movie_id, actor)
        )
        ''')
        self.cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_movie_cast_actor ON movie_cast(actor)
        ''')
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS movie_genres (
            movie_id TEXT,
            genre TEXT,
            PRIMARY KEY (movie_id, genre)
        )
        ''')
        self.cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_movie_genres_genre ON movie_genres(genre)
        ''')

        return

    def prepare(self, fname):
        """Load all records in <staging_area>/<fname> into the SQL database.
        Supports either legacy 'tickets' JSON array or 'movies' JSON array.
        """
        staging_fname = os.path.join(self.staging_area, fname)
        with open(staging_fname, 'r', encoding='utf-8') as file:
            records = json.load(file)
            if not isinstance(records, list):
                raise RuntimeError('Expected a list of objects in staging JSON file.')
            for record in records:
                if 'ticket_id' in record:
                    self._load_ticket(record)
                elif 'movie_id' in record or ('name' in record and 'year' in record):
                    self._load_movie(record)

        self.conn.commit()

        return

    def _load_ticket(self, ticket):
        self.cursor.execute('''
        INSERT INTO tickets (
            ticket_id, request_short, stat_num, status, prio_num, priority,
            date_created, date_closed, date_target, request_type, root_cause,
            main_category, sub_category_1, sub_category_2, owner, editor,
            user_last_name, user_first_name, caller_last_name, caller_first_name,
            request, process, solution
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticket['ticket_id'],
            ticket['request_short'],
            float(ticket['stat_num']),
            ticket['status'],
            float(ticket['prio_num']),
            ticket['priority'],
            ticket['date_created'],
            ticket['date_closed'],
            ticket['date_target'],
            ticket['request_type'],
            ticket['root_cause'],
            ticket['main_category'],
            ticket['sub_category_1'],
            ticket['sub_category_2'],
            ticket['owner'],
            ticket['editor'],
            ticket['user_last_name'],
            ticket['user_first_name'],
            ticket['caller_last_name'],
            ticket['caller_first_name'],
            ticket['request'],
            ticket['process'],
            ticket['solution']
        ))

        return

    def _load_movie(self, movie: dict):
        cast_list = movie.get('cast', [])
        cast_str = ', '.join(cast_list) if isinstance(cast_list, list) else str(cast_list or '')
        genres_list = movie.get('genres', [])
        genres_str = ', '.join(genres_list) if isinstance(genres_list, list) else str(genres_list or '')
        movie_id = movie.get('movie_id') or f"{movie.get('name','').strip()}_{movie.get('year','')}"
        self.cursor.execute('''
        INSERT OR REPLACE INTO movies (
            movie_id, name, year, description, director, cast, genres, rating
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            movie_id,
            movie.get('name', ''),
            int(movie.get('year') or 0),
            movie.get('description', ''),
            movie.get('director', ''),
            cast_str,
            genres_str,
            float(movie.get('rating') or 0.0),
        ))

        # Populate normalized tables
        if isinstance(cast_list, list):
            for actor in cast_list:
                actor_norm = (actor or '').strip()
                if actor_norm:
                    self.cursor.execute('INSERT OR REPLACE INTO movie_cast (movie_id, actor) VALUES (?, ?)', (movie_id, actor_norm))
        if isinstance(genres_list, list):
            for genre in genres_list:
                genre_norm = (genre or '').strip().lower()
                if genre_norm:
                    self.cursor.execute('INSERT OR REPLACE INTO movie_genres (movie_id, genre) VALUES (?, ?)', (movie_id, genre_norm))

        return

    def delete_table(self):
        """Delete all records in the table.
        """
        # Try movies table
        try:
            self.cursor.execute("DELETE FROM movies;")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass
        # Clean normalized tables
        try:
            self.cursor.execute("DELETE FROM movie_cast;")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass
        try:
            self.cursor.execute("DELETE FROM movie_genres;")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass
        # Try legacy tickets table
        try:
            self.cursor.execute("DELETE FROM tickets;")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass

        return

    def close(self):
        """Close the sqlite cursor and connection.
        """
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

        return


class MovieEmbedder:
    def __init__(self, staging_area='', prepared_area='', embedding_store_fname='', embedding_model=''):
        self.staging_area = staging_area
        self.prepared_area = prepared_area
        self.embedding_store_fname = embedding_store_fname
        self.embedding_model = embedding_model

        # Support both Ollama and OpenAI style embedding model ids
        if self.embedding_model.startswith('text-') or self.embedding_model.startswith('text-embedding-'):
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        else:
            self.embeddings = OllamaEmbeddings(model=self.embedding_model)

        return

    def prepare(self, fname):
        """Load JSON data, creates embeddings, and appends it to a vector store.
        """
        staging_fname = os.path.join(self.staging_area, fname)

        loader = JSONLoader(
            file_path=staging_fname,
            jq_schema='.[]',
            text_content=False,
        )

        docs = loader.load()
        prepared_fname = os.path.join(self.prepared_area, self.embedding_store_fname)
        vector_store = SKLearnVectorStore.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_path=prepared_fname,
            serializer='json'
        )
        vector_store.persist()

        return


class MovieGrapher:
    pass


class Prepper:
    """Process the movie files in the staging area and stores the completed files in the prepared area and/or
    in a database. The files processed from the staging area are stored in an archive there.
    """
    def __init__(self, staging_area='', prepared_area='', table_store_fname='', embedding_store_fname='',
                 embedding_model='', graph_url='', graph_username='', graph_password='', verbose=False):
        self.staging_area = staging_area

        self.staging_archive = os.path.join(self.staging_area, 'archive')
        if not os.path.exists(self.staging_archive):
            os.makedirs(self.staging_archive)

        self.tabler_config = {
            'staging_area': staging_area,
            'prepared_area': prepared_area,
            'table_store_fname': table_store_fname
        }
        self.embedder_config = {
            'staging_area': staging_area,
            'prepared_area': prepared_area,
            'embedding_store_fname': embedding_store_fname,
            'embedding_model': embedding_model,
        }
        self.grapher_config = {}

        self.verbose = verbose

        return

    def run(self, delete_table=False, delete_graph=False):
        embedder = MovieEmbedder(**self.embedder_config)

        if delete_table:
            if self.verbose:
                print('Deleting SQL store.')
            with MovieTablerSQL(**self.tabler_config) as tabler:
                tabler.delete_table()

        # Graph functionality removed

        for fname in [f for f in os.listdir(self.staging_area) if f.endswith('.json')]:
            print(f'Prepping: {fname}')
            try:
                if self.verbose:
                    print('..Preparing table store.')
                with MovieTablerSQL(**self.tabler_config) as tabler:
                    tabler.prepare(fname)
                if self.verbose:
                    print('..Preparing embeddings.')
                embedder.prepare(fname)
                # Skip graph preparation

                # Move the file to the staging area's archive.
                staging_fname = os.path.join(self.staging_area, fname)
                archive_fname = os.path.join(self.staging_archive, fname)
                shutil.move(staging_fname, archive_fname)
            except Exception:  # catch all Exceptions
                # Print the error and continue with the rest of the files.
                traceback.print_exc()
                print(f'Error prepping file: {fname}')

        return


if __name__ == '__main__':
    """Instantiate a Prepper and run it once.
    """
    config = utils.get_config(sys.argv[1:])
    prepper_config = {
        'staging_area': config.get('staging_area'),
        'prepared_area': config.get('prepared_area'),
        'table_store_fname': config.get('table_store_fname'),
        'embedding_model': config.get('embedding_model'),
        'embedding_store_fname': config.get('embedding_store_fname'),
        'graph_url': config.get('graph_url'),
        'graph_username': config.get('graph_username'),
        'graph_password': config.get('graph_password'),
        'verbose': config.get('verbose'),
    }
    prepper_run_params = {
        'delete_table': config.get('delete_table', False),
        'delete_graph': config.get('delete_graph', False)
    }

    prepper = Prepper(**prepper_config)
    with timer():
        prepper.run(**prepper_run_params)
