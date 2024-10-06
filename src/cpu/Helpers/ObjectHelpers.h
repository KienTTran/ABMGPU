#ifndef OBJECTHELPERS_H
#define OBJECTHELPERS_H

class ObjectHelpers {
 public:

  template<typename T>
  static void delete_pointer(T *&p) {
    if (p!=nullptr) {
      delete p;
      p = nullptr;
    }
  }

  template<typename T>
  static void clear_vector_memory(std::vector<T *> &vector) {
    if (vector.empty()) return;
    vector.clear();
    std::vector<T *> temp;

    vector.swap(temp);
  }
};

#endif // OBJECTHELPERS_H
