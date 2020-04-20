/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*
 * \file tvm/runtime/data_type.h
 * \brief Primitive runtime data type.
 */
// Acknowledgement: DataType structure design originates from Halide.
#ifndef TVM_RUNTIME_DATA_TYPE_H_
#define TVM_RUNTIME_DATA_TYPE_H_

#include <tvm/runtime/c_runtime_api.h>
#include <dmlc/logging.h>
#include <type_traits>

namespace tvm {
namespace runtime {
/*! \brief Add the support for mixed-bits. */
enum QuanMode {
  RND,
  RND_ZERO,
  RND_MIN_INF,
  RND_INF,
  RND_CONV,
  TRN,
  TRN_ZERO
};
/*! \brief Add the support for overflow mode. */
enum OverMode {
  SAT,
  SAT_ZERO,
  SAT_SYM,
  WRAP,
  WRAP_SM
};

/*!
 * \brief Runtime primitive data type.
 *
 *  This class is a thin wrapper of DLDataType.
 *  We also make use of DataType in compiler to store quick hint
 */
class DataType {
 public:
  /*! \brief Type code for the DataType. */
  enum TypeCode {
    kInt = kDLInt,
    kUInt = kDLUInt,
    kFloat = kDLFloat,
    kHandle = TVMTypeCode::kTVMOpaqueHandle,
  };
  /*! \brief Default constructor initializes
   *         everything to predictable-but-unlikely values
   */
  /*! \brief default constructor */
  DataType() : qmode(RND), omode(SAT) {
    data_.code = 0;
    data_.bits = 0;
    data_.lanes = 0;
    data_.fracs = 0;
  }
  /*!
   * \brief Constructor
   * \param dtype The DLDataType
   */
  explicit DataType(DLDataType dtype)
      : qmode(RND), omode(SAT) {
    data_.code = static_cast<uint8_t>(dtype.code);
    data_.bits = static_cast<uint8_t>(dtype.bits);
    data_.lanes = static_cast<uint16_t>(dtype.lanes);
    data_.fracs = static_cast<uint8_t>(dtype.fracs);
  }
  /*!
   * \brief Constructor
   * \param code The type code.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes.
   */
  DataType(int code, int bits, int lanes,
    int fracs = 0, QuanMode qmode = RND,
    OverMode omode = SAT) {
    data_.code = static_cast<uint8_t>(code);
    data_.bits = static_cast<uint8_t>(bits);
    data_.lanes = static_cast<uint16_t>(lanes);
    data_.fracs = static_cast<uint8_t>(fracs);
    qmode = qmode;
    omode = omode;
  }
  /** \brief trivial copy constructor. */
  DataType(const DataType &that) = default;
  /*! \return The type code. */
  int code() const {
    return static_cast<int>(data_.code);
  }
  /*! \return number of bits in the data. */
  int bits() const {
    return static_cast<int>(data_.bits);
  }
  /*! \return number of bytes to store each scalar. */
  int bytes() const {
    return (bits() + 7) / 8;
  }
  /*! \return the fractional bit size of a single element of this type. */
  int fracs() const {
    return static_cast<int>(data_.fracs);
  }
  /*! \return number of lanes in the data. */
  int lanes() const {
    return static_cast<int>(data_.lanes);
  }
  /*! \return whether type is a scalar type. */
  bool is_scalar() const {
    return lanes() == 1;
  }
  /*! \return whether type is a scalar type. */
  bool is_bool() const {
    return code() == DataType::kUInt && bits() == 1;
  }
  /*! \return whether type is a float type. */
  bool is_float() const {
    return code() == DataType::kFloat;
  }
  /*! \return whether type is a float16 type. */
  bool is_float16() const {
    return is_float() && bits() == 16;
  }
  /*! \return whether type is an int type. */
  bool is_int() const {
    return code() == DataType::kInt;
  }
  /*! \return whether type is an uint type. */
  bool is_uint() const {
    return code() == DataType::kUInt;
  }
  /** Is this type an signed fixed-point type? */
  bool is_fixed() const {
    return code() == kDLInt && fracs() >= 0;
  }
  /** Is this type an unsigned fixed-point type? */
  bool is_ufixed() const {
    return code() == kDLUInt && fracs() >= 0;
  }
  /*! \return whether type is a handle type. */
  bool is_handle() const {
    return code() == DataType::kHandle;
  }
  /*! \return whether type is a vector type. */
  bool is_vector() const {
    return lanes() > 1;
  }
  /*! \return DataType with same number of bits and lanes,
   * but new_code for a type code.
   */
  DataType with_code(int new_code) const {
      return DataType(new_code, bits(), lanes(),
        fracs(), qmode, omode);
  }
  /*!
   * \brief Create a new data type by change lanes to a specified value.
   * \param lanes The target number of lanes.
   * \return the result type.
   */
  DataType with_lanes(int lanes) const {
    return DataType(data_.code, data_.bits, lanes,
      data_.fracs, qmode, omode);
  }
  /*!
   * \brief Create a new data type by change bits to a specified value.
   * \param bits The target number of bits.
   * \return the result type.
   */
  DataType with_bits(int bits) const {
    return DataType(data_.code, bits, data_.lanes,
      data_.fracs, qmode, omode);
  }
  /*! \return DataType with same type code and lanes, but new_fracs for
   *  the number of fracs  bits.
   */
  DataType with_fracs(int new_fracs) const {
      return DataType(data_.code, data_.bits, data_.lanes,
        new_fracs, qmode, omode);
  }
  /*!
   * \brief Get the scalar version of the type.
   * \return the result type.
   */
  DataType element_of() const {
    return with_lanes(1);
  }
  /*!
   * \brief Equal comparator.
   * \param other The data type to compre against.
   * \return The comparison resilt.
   */
  bool operator==(const DataType& other) const {
    return
        data_.code == other.data_.code &&
        data_.bits == other.data_.bits &&
        data_.lanes == other.data_.lanes &&
        qmode == other.qmode &&
        omode == other.omode;
  }
  /*!
   * \brief NotEqual comparator.
   * \param other The data type to compre against.
   * \return The comparison resilt.
   */
  bool operator!=(const DataType& other) const {
    return !operator==(other);
  }
  /*!
   * \brief Converter to DLDataType
   * \return the result.
   */
  operator DLDataType () const {
    return data_;
  }

  /*!
   * \brief Construct an int type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes.
   * \return The constructed data type.
   */
  static DataType Int(int bits, int lanes = 1, int fracs = 0) {
    return DataType(kDLInt, bits, lanes, fracs);
  }
  /*!
   * \brief Construct an uint type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType UInt(int bits, int lanes = 1, int fracs = 0) {
    return DataType(kDLUInt, bits, lanes, fracs);
  }
  /*!
   * \brief Construct an uint type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Float(int bits, int lanes = 1) {
    return DataType(kDLFloat, bits, lanes);
  }
  /*!
   * \brief Construct a bool type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Bool(int lanes = 1) {
    return DataType::UInt(1, lanes);
  }
  /*!
   * \brief Construct a handle type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Handle(int bits = 64, int lanes = 1) {
    return DataType(kHandle, bits, lanes, 0, RND, SAT);
  }
  /** Constructing a signed fixed-point type */
  inline DataType Fixed(int bits, int fracs, int lanes = 1,
    QuanMode qmode = RND, OverMode omode = SAT) {
      return DataType(kDLInt, bits, lanes, fracs,
        qmode, omode);
  }
  /** Constructing a unsigned fixed-point type */
  inline DataType UFixed(int bits, int fracs, int lanes = 1,
    QuanMode qmode = RND, OverMode omode = SAT) {
      return DataType(kDLUInt, bits, lanes, fracs,
        qmode, omode);
  }
  /*!
   * \brief Get the corresponding type of TVMShapeIndex.
   * \return The type of TVM shape index.
   */
  static DataType ShapeIndex() {
    if (std::is_signed<tvm_index_t>::value) {
      return DataType::Int(sizeof(tvm_index_t) * 8);
    } else {
      return DataType::UInt(sizeof(tvm_index_t) * 8);
    }
  }

 private:
  DLDataType data_;
  /*! \brief Add the support for mixed-bits. */
  QuanMode qmode;
  /*! \brief Add the support for overflow mode. */
  OverMode omode;
};

/*!
 * \brief Get the number of bytes needed in a vector.
 * \param dtype The data type.
 * \return Number of bytes needed.
 */
inline int GetVectorBytes(DataType dtype) {
  int data_bits = dtype.bits() * dtype.lanes();
  // allow bool to exist
  if (dtype == DataType::Bool() ||
      dtype == DataType::Int(4) ||
      dtype == DataType::UInt(4) ||
      dtype == DataType::Int(1)) {
    return 1;
  }
  // CHECK_EQ(data_bits % 8, 0U)
  //     << "Need to load/store by multiple of bytes";
  return (data_bits + 7) / 8;
}

/*!
 * \brief Check whether type matches the given spec.
 * \param t The type
 * \param code The type code.
 * \param bits The number of bits to be matched.
 * \param lanes The number of lanes in the type.
 */
inline bool TypeMatch(DLDataType t, int code, int bits, int lanes = 1) {
  return t.code == code && t.bits == bits && t.lanes == lanes;
}
/*!
 * \brief Check whether two types are equal .
 * \param lhs The left operand.
 * \param rhs The right operand.
 */
inline bool TypeEqual(DLDataType lhs, DLDataType rhs) {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}
}  // namespace runtime

using DataType = runtime::DataType;

}  // namespace tvm
#endif  //  TVM_RUNTIME_DATA_TYPE_H_
