import {FastifyReply, FastifyRequest} from "fastify";
import bcrypt from "bcrypt";


export async function addUser(req: FastifyRequest, res: FastifyReply) {
    const { name, email, password } = req.body as { name: string; email: string; password: string };

    if (!name || !email || !password) {
        return res.status(400).send({ error: "Toutes les informations sont requises !" });
    }
    try {
        const response = await fetch(`http://database:8003/email-existing?email=${encodeURIComponent(email)}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        if (!response.ok) {
            return res.status(400).send({ error: "Email déjà utilisé" });
        }
        const hashedPassword = await bcrypt.hash(password, 10);
        const addUserRes = await fetch('http://database:8003/addUser', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: name,
                email: email,
                password: hashedPassword
            })
        });
        if (!addUserRes.ok) {
            return res.status(500).send({ error: "something come wrong" });
        }
        return res.status(201).send();
    } catch (err) {
        console.error(err);
        return false;
    }

}