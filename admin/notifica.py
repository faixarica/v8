
import streamlit as st
import smtplib

from datetime import date, timedelta
from sqlalchemy import text
from db import Session

def processar_notificacoes_acertos():
    """
    Verifica os palpites do dia anterior, compara com o resultado oficial
    e envia notificações automáticas para os usuários.
    """
    st.markdown("### 🚀 Processar Notificações de Acertos")

    db = Session()
    ontem = date.today() - timedelta(days=1)

    try:
        # 1️⃣ Buscar resultado oficial de ontem
        res = db.execute(text("""
            SELECT n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,
                   n11,n12,n13,n14,n15
            FROM resultados_oficiais
            WHERE data = :data
        """), {"data": ontem}).fetchone()

        if not res:
            st.warning(f"Nenhum resultado oficial encontrado para {ontem}.")
            return

        resultado = set(res)

        # 2️⃣ Buscar palpites do dia anterior
        palpites = db.execute(text("""
            SELECT p.id, p.id_usuario, p.numeros, u.email, u.nome
            FROM palpites p
            JOIN usuarios u ON u.id = p.id_usuario
            WHERE DATE(p.data) = :data
              AND p.id NOT IN (SELECT id_palpite FROM notificacoes_palpite)
        """), {"data": ontem}).fetchall()

        if not palpites:
            st.info("Nenhum palpite pendente para notificação.")
            return

        enviados = 0
        for palpite in palpites:
            numeros = {int(x) for x in palpite.numeros.split(',')}
            acertos = len(resultado.intersection(numeros))

            if acertos >= 11:
                msg = (
                    f"🎉 Parabéns, {palpite.nome}!\n"
                    f"Seu palpite do dia {ontem.strftime('%d/%m/%Y')} "
                    f"acertou {acertos} números na Lotofácil! 🍀"
                )

                # 3️⃣ Enviar e-mail (exemplo simples, depois pode trocar por API)
                enviar_email(palpite.email, "Faixabet - Resultado do seu palpite", msg)

                # 4️⃣ Registrar notificação
                db.execute(text("""
                    INSERT INTO notificacoes_palpite (id_palpite, id_usuario, acertos, canal, mensagem)
                    VALUES (:pid, :uid, :acertos, :canal, :msg)
                """), {
                    "pid": palpite.id,
                    "uid": palpite.id_usuario,
                    "acertos": acertos,
                    "canal": "email",
                    "msg": msg
                })
                enviados += 1

        db.commit()
        st.success(f"✅ {enviados} notificações enviadas com sucesso!")

    except Exception as e:
        st.error(f"Erro ao processar notificações: {e}")
        db.rollback()
    finally:
        db.close()


def enviar_email(destinatario, assunto, corpo):
    """Função simples para envio de e-mail (pode trocar por SendGrid, SMTP etc.)."""
    try:
        servidor = smtplib.SMTP("smtp.gmail.com", 587)
        servidor.starttls()
        servidor.login("faixaricaa@gmail.com", "senha_app")
        mensagem = f"Subject: {assunto}\n\n{corpo}"
        servidor.sendmail("faixaricaa@gmail.com", destinatario, mensagem)
        servidor.quit()
    except Exception as e:
        print(f"Erro ao enviar e-mail para {destinatario}: {e}")
