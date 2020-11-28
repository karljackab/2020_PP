ssh-keygen -t rsa
for i in {1..9};do
    cat ~/.ssh/id_rsa.pub | ssh "pp${i}" 'mkdir -p ~/.ssh; cat >> .ssh/authorized_keys'
done