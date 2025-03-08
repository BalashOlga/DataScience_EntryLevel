class SQL_connection:
    def __init__(self, url):
        # Создание движка
        from sqlalchemy import create_engine
        self.engine = create_engine(url)
       
        # Подключение к базе данных
        try:
            self.connection = self.engine.connect()  # Используем self.engine
            print("Подключение к базе данных успешно установлено!")
        except Exception as e:
            print(f"Ошибка: {e}")

    def close(self):
        try:
            self.connection.close()  # Используем self.connection
            print('Соединение с базой данных закрыто')
        except Exception as e:
            print(f"Ошибка: {e}")