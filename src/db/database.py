from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


class Database:
    def __init__(self, db_url: str):
        self.engine = create_engine(
            db_url,
            pool_pre_ping=True
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    def _get_session(self):
        return self.SessionLocal()

    def create_job(
        self,
        image_url: str,
        model_name: str = "deeplabv3_resnet50"
    ) -> int:
        session = self._get_session()
        try:
            query = text("""
                INSERT INTO segmentation_jobs (
                    image_url,
                    model_name,
                    status
                )
                VALUES (
                    :image_url,
                    :model_name,
                    'processing'
                )
                RETURNING id
            """)
            result = session.execute(query, {
                "image_url": image_url,
                "model_name": model_name
            })
            session.commit()
            return result.scalar()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def complete_job(
        self,
        job_id: int,
        mask_url: str,
        overlay_url: str,
        inference_time_ms: int
    ):
        session = self._get_session()
        try:
            query = text("""
                UPDATE segmentation_jobs
                SET
                    mask_url = :mask_url,
                    overlay_url = :overlay_url,
                    inference_time_ms = :inference_time_ms,
                    status = 'completed'
                WHERE id = :job_id
            """)
            session.execute(query, {
                "job_id": job_id,
                "mask_url": mask_url,
                "overlay_url": overlay_url,
                "inference_time_ms": inference_time_ms
            })
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
