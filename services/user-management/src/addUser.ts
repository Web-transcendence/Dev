import {FastifyReply, FastifyRequest} from "fastify";
import jwt from 'jsonwebtoken';

export async function addUser(req: FastifyRequest, res: FastifyReply):Promise<void> {
    const { name, email, password } = req.body as { name: string; email: string; password: string };
    try {
        const response = await fetch(`http://database:4001/email-existing?email=${encodeURIComponent(email)}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        if (!response.ok) {
            return res.status(400).send({ error: "Email déjà utilisé" });
        }
        console.log(response);
        console.log("CACACACACACACACAC");
        const addUserRes = await fetch('http://database:4001/addUser', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: name,
                email: email,
                password: password
                // password: hashedPassword
            })
        });
        console.log("CACACACACA11111111111CACACAC");
        if (!addUserRes.ok) {
            const errorData = await addUserRes.json();
            return res.status(500).send({ json: errorData});
        }
        console.log("CACACACA222222222CACACACAC");
        const user = { id: 1, name: name };
        const token = jwt.sign(user, 'secret_key', { expiresIn: '1h' });
        res.status(201).send({token, username: name, redirect: 'post/login'});
    } catch (err) {
        console.log("CACACACACA33333333CAC");
        console.error(err);
    }
}